"""Conversation state machine: Capture (this module) -> Explain & Secure (ASIDE).

States per session:
  COLLECTING   -- still asking clarifying questions
  PRESENTING   -- a ranked shortlist is on screen; the user can buy one or
                  more of the shown items (in any order, one at a time or
                  several in the same message), keep asking about others,
                  or start a new search entirely. This stage does NOT end
                  the session -- there used to be a GATED/DONE dead end here
                  that made the agent say "this session has ended" after a
                  single purchase evaluation. That was a bug, not a design
                  choice: a real shopper checks several things before
                  deciding, and nothing about the security gate requires
                  ending the conversation once it's used once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app import anomaly as anomaly_module, audit_log, payments
from app.aside_client import GateResult, evaluate
from app.behavioral_profile import DualLayerOutput, build_dual_layer
from app.counterfactual import CounterfactualResult, assess_choice as assess_cdc, rederive as rederive_counterfactual
from app.decision_object import DecisionTrajectory
from app.intent import IntentSlots, extract_intent, next_clarifying_question
from app.irreversibility import classify as classify_irreversibility
from app.retrieval import RankedProduct, retrieve_and_rank
from app.review_authenticity import check as check_review_authenticity

ID_RE = re.compile(r"\b([a-z]-\d{3,6})\b", re.I)
RESET_RE = re.compile(
    r"\b(new search|start over|restart|something else|browse again|different (category|search))\b", re.I
)
APPROVE_RE = re.compile(r"\b(approve|reject|deny)\b.*?\b([a-z]-\d{3,6})\b", re.I)

# Decisions ASIDE can return, ranked from least to most restrictive. Used by
# the irreversibility escalation below: an ALLOW is upgraded, a DENY never
# gets downgraded.
DECISION_RANK = {"ALLOW": 0, "REQUIRE_HUMAN_APPROVAL": 1, "REQUIRE_VERIFIABLE_APPROVAL": 2, "DENY": 3}

# Maps a clarifying question's text back to the slot it was asking for, so a
# bare "3000" answer can be trusted as a budget only when that's actually
# what was just asked (see app/intent.py's `expecting` parameter).
QUESTION_TO_SLOT = {
    "What kind of product are we looking at (e.g. jewellery, electronics, a DIY project)?": "category",
    "What's your budget, roughly (in INR)?": "budget_max",
}


@dataclass
class SessionState:
    session_id: str
    agent_id: str
    user_id: str
    slots: IntentSlots | None = None
    questions_asked: int = 0
    stage: str = "COLLECTING"
    ranked: list[RankedProduct] = field(default_factory=list)
    trajectory: DecisionTrajectory | None = None
    # Counterfactual Decision Checker result for the shortlist currently on
    # screen (see app/counterfactual.py) -- recomputed every time a new
    # shortlist is retrieved, read by _evaluate_one to decide whether the
    # item about to be bought was the shortlist's "won only on gameable
    # signals" top pick.
    counterfactual: CounterfactualResult | None = None
    # Simple per-session counters so the demo can show the security gate
    # actually react to velocity / new-customer signals live, without a
    # real user-account database behind it.
    recent_transaction_count: int = 0
    historical_transaction_count: int = 0
    # product_id -> {"chosen": RankedProduct, "gate": GateResult, "dual_layer": DualLayerOutput}
    # for every purchase currently sitting at REQUIRE_HUMAN_APPROVAL /
    # REQUIRE_VERIFIABLE_APPROVAL, waiting on the "additional process" a
    # flagged purchase needs: something to actually resolve it, instead of
    # the conversation just stopping at the flag. See "approve <id>" /
    # "reject <id>" handling in handle_turn.
    pending_approvals: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)  # product_id -> executed order


@dataclass
class TurnResult:
    reply: str
    stage: str
    products: list[dict] = field(default_factory=list)
    gate: GateResult | None = None  # most recent evaluation this turn, for backward compatibility
    gates: list[dict] = field(default_factory=list)  # every evaluation THIS turn (supports multi-id purchases)
    trajectory: dict | None = None
    dual_layer: dict | None = None  # most recent evaluation's dual-layer output
    # True exactly on the turn that handled a "new search"/"start over" reset.
    # Bug fix: the front end had no way to know a reset happened, so it kept
    # showing the previous shortlist and chat log on screen even after the
    # backend had cleared its own slots/stage. static/app.js clears its UI
    # when this is true.
    restarted: bool = False


def _product_card(r: RankedProduct) -> dict:
    p = r.product
    return {
        "id": p.id, "name": p.name, "price": p.price, "category": p.category,
        "merchant": p.merchant, "is_new_merchant": p.is_new_merchant,
        "justification": r.justification, "delivery_days": p.delivery_days,
    }


def _collect_turn(state: SessionState, message: str, *, is_restart: bool = False) -> TurnResult:
    """The COLLECTING-stage logic: fill slots turn by turn, ask at most
    MAX_CLARIFYING_QUESTIONS clarifying questions, then retrieve and rank.
    Pulled into its own function so a mid-session "new search" (see
    RESET_RE) can re-enter it without duplicating the logic.
    """
    if is_restart:
        state.slots = None
        state.questions_asked = 0
        state.ranked = []
        state.counterfactual = None
        state.stage = "COLLECTING"

    if state.slots is None:
        state.slots = extract_intent(message)
    else:
        # This message is answering the last clarifying question.
        last_q = getattr(state, "_last_question", None)
        expecting = QUESTION_TO_SLOT.get(last_q)
        state.slots = extract_intent(message, existing=state.slots, expecting=expecting)
        if last_q:
            state.trajectory.record_clarifying(last_q, message)

    state.trajectory.record_slots(state.slots)

    question = next_clarifying_question(state.slots, state.questions_asked)
    if question:
        state.questions_asked += 1
        state._last_question = question  # type: ignore[attr-defined]
        return TurnResult(reply=question, stage=state.stage, restarted=is_restart)

    if state.slots.category is None:
        # The clarifying-question budget is spent and we STILL don't know
        # what category they want -- e.g. every answer was unrecognized
        # input like "a". Retrieving with category=None would return the
        # entire catalogue (a real bug: it looked like 5 matching results
        # for a request the agent never actually understood). Say so
        # honestly instead, and reset so the next message gets a clean
        # attempt rather than continuing to count against an exhausted cap.
        state.slots = None
        state.questions_asked = 0
        state.stage = "COLLECTING"
        return TurnResult(
            reply=(
                "I still couldn't tell what kind of product you're looking for, so I don't have a match -- "
                "try naming it directly, e.g. \"electronics, budget 5000\"."
            ),
            stage=state.stage,
            restarted=is_restart,
        )

    # Enough slots resolved (or clarifying budget spent) -- retrieve.
    state.ranked = retrieve_and_rank(state.slots)
    rationale = (
        f"Filtered catalogue to category={state.slots.category!r}, "
        f"budget<=₹{state.slots.budget_max:,.0f}" if state.slots.budget_max else
        f"Filtered catalogue to category={state.slots.category!r}"
    )
    state.trajectory.record_retrieval(state.ranked, rationale)
    state.stage = "PRESENTING"

    if not state.ranked:
        # Deliberately NOT a dead end: stay in COLLECTING so the user can
        # just describe a different situation on their next message.
        state.stage = "COLLECTING"
        state.slots = None
        state.questions_asked = 0
        from app import catalog as _catalog
        if not _catalog.CATALOGUE:
            return TurnResult(reply=("The real product catalogue hasn't been built yet, so there is nothing to buy. "
                                     "Run scripts/build_catalog.py (see DATA.md)."),
                              stage=state.stage, restarted=is_restart)
        return TurnResult(reply="No products matched that -- try describing a different budget or category.",
                           stage=state.stage, restarted=is_restart)

    # Counterfactual Decision Checker (app/counterfactual.py): re-derive this
    # shortlist's #1 spot from trusted-only signals and compare against what
    # was actually surfaced, BEFORE the shopper has picked anything. This is
    # a check on the shortlist itself, so it's recomputed here, once per
    # retrieval, rather than per purchase.
    state.counterfactual = rederive_counterfactual(state.ranked, state.slots)
    state.trajectory.record_counterfactual(state.counterfactual.as_dict())

    reply = f"Here's what I'd suggest ({len(state.ranked)} options), each with why:"
    return TurnResult(reply=reply, stage=state.stage,
                       products=[_product_card(r) for r in state.ranked],
                       trajectory=state.trajectory.as_dict(),
                       restarted=is_restart)


def _listing_text(product) -> str:
    """The free text an outside party controls (title, description, review bodies). It MUST reach the gate: a poisoned
    listing hides its instructions here, and the old fixed 'product data returned normally' string meant ASIDE could
    never see them."""
    reviews = " ".join(str(r.get("text", "")) for r in (product.reviews or [])[:8])
    return f"{product.name}. {product.description} {reviews}".strip()[:6000]


def _evaluate_one(state: SessionState, chosen: RankedProduct) -> tuple[GateResult, DualLayerOutput]:
    # Buyer-side evidence is computed BEFORE the gate so ASIDE can act on it (and it lands in ASIDE's audit record):
    irreversibility = classify_irreversibility(chosen.product, amount=chosen.product.price)
    anomaly_result = anomaly_module.assess_product(chosen.product)
    cdc_v2 = assess_cdc(state.ranked, state.slots, chosen.product.id, irreversibility["score"], irreversibility["level"])
    # CDC asks "was the AGENT's decision steered?". When the shopper explicitly picks an item that is not the
    # agent's own top recommendation, that is the user's choice, not an agent decision: the divergence is kept for
    # display but neither sent as a risk signal nor escalated.
    user_picked = bool(state.ranked) and state.ranked[0].product.id != chosen.product.id
    informational_w = cdc_v2.weighted_divergence
    if user_picked:
        cdc_v2.weighted_divergence, cdc_v2.level = 0.0, "NONE"
        cdc_v2.note = ("The shopper explicitly chose an item other than the agent's top recommendation, so divergence "
                       f"from the clean re-derivation (informational: {informational_w:.2f}) is a preference, not a manipulation signal.")
    review_authenticity = check_review_authenticity(chosen.product)
    agentra_signals = {
        "cdc": {"weighted_divergence": cdc_v2.weighted_divergence, "divergence": cdc_v2.divergence,
                "semantic_distance": cdc_v2.semantic_distance, "level": cdc_v2.level,
                "user_selected": user_picked},
        "irreversibility": {"score": irreversibility["score"], "level": irreversibility["level"]},
        "anomaly": {"score": anomaly_result["score"], "level": anomaly_result["level"]},
        "review_authenticity": {"suspicious": review_authenticity.suspicious},
    }
    gate_payload = {
        "agent_id": state.agent_id,
        "user_id": state.user_id,
        "action": "purchase",
        "tool": "commerce.checkout",
        "merchant": chosen.product.merchant,
        "amount": chosen.product.price,
        "currency": "INR",
        "is_new_merchant": chosen.product.is_new_merchant,
        "historical_transaction_count": state.historical_transaction_count,
        "recent_transaction_count": state.recent_transaction_count,
        "tool_output": _listing_text(chosen.product),
        "event_type": "commerce_purchase",
        "metadata": {
            "category": chosen.product.category,
            "channel": "web",
            "agentra": agentra_signals,
        },
    }
    gate = evaluate(gate_payload)
    state.recent_transaction_count += 1
    state.historical_transaction_count += 1

    # Fix: ASIDE's own guardrails escalate on amount and merchant novelty,
    # but nothing was reading the irreversibility classification at all --
    # it was computed only for display, after the decision was already
    # final. That meant a merchant-established, under-threshold but
    # permanent/personalized item (e.g. an engraved ring) could sail through
    # as a plain ALLOW with no human ever looking at it. This is Agentra's
    # own layer applying the same "guardrails override the model" principle
    # ASIDE uses internally, one level up: an irreversibility read of
    # HIGH/CRITICAL raises the floor to at least REQUIRE_HUMAN_APPROVAL,
    # never below what ASIDE itself already decided.
    if (irreversibility["level"] in ("HIGH", "CRITICAL")
            and DECISION_RANK.get(gate.decision, 0) < DECISION_RANK["REQUIRE_HUMAN_APPROVAL"]):
        original_decision = gate.decision
        gate.decision = "REQUIRE_HUMAN_APPROVAL"
        gate.risk_factors = list(gate.risk_factors) + ["agentra_irreversibility_escalation"]
        gate.reasoning = (
            f"Agentra escalated ASIDE's {original_decision} to REQUIRE_HUMAN_APPROVAL: irreversibility "
            f"{irreversibility['level']} ({irreversibility['score']}/100) -- {'; '.join(irreversibility['factors'])}. "
            f"[Underlying ASIDE reasoning: {gate.reasoning}]"
        )

    # Fix / feature: the Counterfactual Decision Checker (app/counterfactual.py)
    # was computed for display but nothing acted on it either -- the same gap
    # the irreversibility escalation above closes. If this specific item is
    # the shortlist's top pick AND that top spot only holds once gameable
    # signals (review score, brand tag) are counted -- exactly the shape of a
    # poisoned/spoofed listing -- raise the floor the same way, on top of
    # whatever irreversibility already did. A DENY from ASIDE is still never
    # downgraded.
    if (state.counterfactual and state.counterfactual.diverged
            and state.counterfactual.actual_top_id == chosen.product.id
            and DECISION_RANK.get(gate.decision, 0) < DECISION_RANK["REQUIRE_HUMAN_APPROVAL"]):
        original_decision = gate.decision
        gate.decision = "REQUIRE_HUMAN_APPROVAL"
        gate.risk_factors = list(gate.risk_factors) + ["agentra_counterfactual_escalation"]
        gate.reasoning = (
            f"Agentra escalated ASIDE's {original_decision} to REQUIRE_HUMAN_APPROVAL: counterfactual "
            f"re-derivation diverged from the shown shortlist -- {state.counterfactual.note} "
            f"[Underlying ASIDE reasoning: {gate.reasoning}]"
        )

    # Review Authenticity check (Anomaly Detection Module, app/review_authenticity.py):
    # independent of everything above -- this looks at whether the item's
    # OWN review data looks organic, not at how the agent ranked it. A
    # fabricated 5-star average can slip past a perfectly fair ranking
    # algorithm; catching it needs to look at the reviews themselves.
    if (review_authenticity.suspicious
            and DECISION_RANK.get(gate.decision, 0) < DECISION_RANK["REQUIRE_HUMAN_APPROVAL"]):
        original_decision = gate.decision
        gate.decision = "REQUIRE_HUMAN_APPROVAL"
        gate.risk_factors = list(gate.risk_factors) + ["agentra_review_authenticity_escalation"]
        gate.reasoning = (
            f"Agentra escalated ASIDE's {original_decision} to REQUIRE_HUMAN_APPROVAL: this listing's "
            f"reviews look fabricated -- {' '.join(review_authenticity.reasons)} "
            f"[Underlying ASIDE reasoning: {gate.reasoning}]"
        )

    # Defence in depth: ASIDE (any version) is asked to act on these signals, but if it predates them or is
    # unreachable we still never let a flagged purchase through as a plain ALLOW.
    for name, hit, why in (
        ("agentra_cdc_v2_escalation", cdc_v2.level in ("ESCALATE", "BLOCK"), cdc_v2.note),
        ("agentra_anomaly_escalation", anomaly_result["level"] == "HIGH", " ".join(anomaly_result["reasons"])),
    ):
        if hit and DECISION_RANK.get(gate.decision, 0) < DECISION_RANK["REQUIRE_HUMAN_APPROVAL"]:
            original_decision = gate.decision
            gate.decision = "REQUIRE_HUMAN_APPROVAL"
            gate.risk_factors = list(gate.risk_factors) + [name]
            gate.reasoning = (f"Agentra escalated ASIDE's {original_decision} to REQUIRE_HUMAN_APPROVAL: {why} "
                              f"[Underlying ASIDE reasoning: {gate.reasoning}]")
    gate.agentra_signals = {**agentra_signals, "cdc_assessment": cdc_v2.as_dict(), "anomaly_detail": anomaly_result}

    state.trajectory.record_final_selection(chosen)
    state.trajectory.record_gate(gate)

    # Dual-layer output (Patent Claim 2): one attribution pass over the
    # chosen product + shortlist, rendered as both a consumer explanation
    # and an enterprise behavioral profile. Computed regardless of the
    # gate's decision -- explaining a DENY or an escalation matters as much
    # as explaining an ALLOW.
    dual_layer = build_dual_layer(chosen=chosen, slots=state.slots, ranked_shortlist=state.ranked,
                                   user_id=state.user_id, counterfactual=state.counterfactual)
    state.trajectory.record_dual_layer(dual_layer)
    state.trajectory.record_purchase_evaluation(chosen.product.id, chosen.product.name, gate, dual_layer)

    # The "additional process" a flagged purchase needs: park it as pending
    # rather than just displaying the flag and moving on, so there's
    # somewhere for a human reviewer's actual decision to go.
    if gate.decision in ("REQUIRE_HUMAN_APPROVAL", "REQUIRE_VERIFIABLE_APPROVAL"):
        state.pending_approvals[chosen.product.id] = {"chosen": chosen, "gate": gate, "dual_layer": dual_layer}
    else:
        state.pending_approvals.pop(chosen.product.id, None)

    return gate, dual_layer


def _settle(state: SessionState, chosen: RankedProduct, gate: GateResult) -> str:
    """Execute the purchase after an ALLOW (direct or reviewer-approved) and
    write the audit trail. Returns a sentence fragment for the reply."""
    p = chosen.product
    audit_log.append("gate_decision", {"session_id": state.session_id, "product_id": p.id,
                                       "decision": gate.decision, "risk_score": gate.risk_score,
                                       "source": p.source})
    try:
        order = payments.execute_purchase(state.session_id, p.id, p.name, p.price)
    except payments.PaymentRefused as exc:
        audit_log.append("purchase_refused", {"session_id": state.session_id, "product_id": p.id, "reason": str(exc)})
        return f"{p.name}: allowed by the gate but not purchased -- {exc}."
    state.orders[p.id] = order
    state.recent_transaction_count += 1
    audit_log.append("purchase_executed", {"session_id": state.session_id, "product_id": p.id, **order})
    label = "simulated order" if order["mode"] == "dry_run" else "Razorpay test-mode order"
    return f"{p.name}: allowed and purchased ({label} {order['order_id']}, ₹{p.price:,.2f})."


def _explain_factors(gate: GateResult) -> str:
    """Surface WHY, beyond `reasoning` alone. ASIDE's `reasoning` string can
    be generic (e.g. its own demo-mode fallback text), but `violated_policies`
    and `risk_factors` usually name the actual triggering rule (a velocity
    limit, a new-merchant check, etc.) even when the prose doesn't. Showing
    both means a confusing generic reasoning string isn't the only thing the
    shopper sees."""
    named = list(dict.fromkeys((gate.violated_policies or []) + (gate.risk_factors or [])))
    return f" [{', '.join(named)}]" if named else ""


def _decision_sentence(product_id: str, product_name: str, gate: GateResult) -> str:
    if gate.decision == "ALLOW":
        return f"{product_name}: allowed."
    if gate.decision == "DENY":
        return f"{product_name}: blocked before payment -- {gate.reasoning}{_explain_factors(gate)}"
    # REQUIRE_HUMAN_APPROVAL / REQUIRE_VERIFIABLE_APPROVAL: the flag by
    # itself used to be a dead end -- nothing let the conversation actually
    # resolve it. Now it names the concrete next step.
    return (f"{product_name}: flagged for {gate.decision.replace('_', ' ').title()}{_explain_factors(gate)}. "
            f"Reply \"approve {product_id}\" or \"reject {product_id}\" to simulate the human reviewer's call.")


def handle_turn(state: SessionState, message: str) -> TurnResult:
    if state.trajectory is None:
        state.trajectory = DecisionTrajectory.start(
            state.session_id, state.agent_id, state.user_id, stated_intent=message
        )

    if state.stage == "COLLECTING":
        return _collect_turn(state, message)

    if state.stage == "PRESENTING":
        if RESET_RE.search(message):
            return _collect_turn(state, message, is_restart=True)

        approval_match = APPROVE_RE.search(message)
        if approval_match:
            verb, pid = approval_match.group(1).lower(), approval_match.group(2).lower()
            pending = state.pending_approvals.get(pid)
            if pending is None:
                return TurnResult(
                    reply=f"{pid} isn't waiting on an approval right now.",
                    stage=state.stage, products=[_product_card(r) for r in state.ranked],
                )
            resolved_decision = "ALLOW" if verb == "approve" else "DENY"
            product_name = pending["chosen"].product.name
            resolution_note = (
                f"Resolved by human reviewer: {verb}d following {pending['gate'].decision.replace('_', ' ').title()}."
            )
            resolved_gate = GateResult(
                decision=resolved_decision, risk_score=pending["gate"].risk_score,
                confidence=pending["gate"].confidence, threats=pending["gate"].threats,
                violated_policies=pending["gate"].violated_policies,
                risk_factors=pending["gate"].risk_factors, reasoning=resolution_note,
                evidence=pending["gate"].evidence, request_id=pending["gate"].request_id,
            )
            state.trajectory.record_gate(resolved_gate)
            state.trajectory.record_purchase_evaluation(pid, product_name, resolved_gate, pending["dual_layer"])
            del state.pending_approvals[pid]
            if resolved_decision == "ALLOW":
                reply = f"Reviewer approved. {_settle(state, pending['chosen'], resolved_gate)}"
            else:
                audit_log.append("reviewer_rejected", {"session_id": state.session_id, "product_id": pid})
                reply = f"{product_name}: reviewer rejected it -- purchase stays blocked."
            return TurnResult(
                reply=reply, stage=state.stage, products=[_product_card(r) for r in state.ranked],
                gate=resolved_gate,
                gates=[{"product_id": pid, "product_name": product_name, "decision": resolved_decision,
                        "risk_score": resolved_gate.risk_score, "confidence": resolved_gate.confidence,
                        "reasoning": resolution_note, "unreachable": False}],
                trajectory=state.trajectory.as_dict(),
            )

        ids_in_message = []
        for m in ID_RE.finditer(message):
            pid = m.group(1).lower()
            if pid not in ids_in_message:
                ids_in_message.append(pid)

        if not ids_in_message:
            return TurnResult(
                reply=(
                    "Tell me which one(s) to check (e.g. \"buy j-104\" or \"check j-104 and e-201\"), "
                    "or say \"new search\" to look at something else."
                ),
                stage=state.stage,
                products=[_product_card(r) for r in state.ranked],
            )

        shortlist_by_id = {r.product.id: r for r in state.ranked}
        valid_ids = [pid for pid in ids_in_message if pid in shortlist_by_id]
        invalid_ids = [pid for pid in ids_in_message if pid not in shortlist_by_id]

        if not valid_ids:
            return TurnResult(
                reply=f"{', '.join(invalid_ids)} wasn't in the shortlist -- pick one of the ids shown.",
                stage=state.stage, products=[_product_card(r) for r in state.ranked],
            )

        gates: list[dict] = []
        sentences = []
        last_gate: GateResult | None = None
        last_dual_layer: DualLayerOutput | None = None

        for pid in valid_ids:
            chosen = shortlist_by_id[pid]
            gate, dual_layer = _evaluate_one(state, chosen)
            last_gate, last_dual_layer = gate, dual_layer
            if gate.decision == "ALLOW":
                sentences.append(_settle(state, chosen, gate))
            else:
                audit_log.append("gate_decision", {"session_id": state.session_id, "product_id": pid,
                                                   "decision": gate.decision, "risk_score": gate.risk_score})
                sentences.append(_decision_sentence(pid, chosen.product.name, gate))
            gates.append({
                "product_id": pid,
                "product_name": chosen.product.name,
                "decision": gate.decision,
                "risk_score": gate.risk_score,
                "confidence": gate.confidence,
                "reasoning": gate.reasoning,
                "unreachable": gate.unreachable,
            })

        if invalid_ids:
            sentences.append(f"(ignored -- not in the shortlist: {', '.join(invalid_ids)})")

        reply = " ".join(sentences)
        # Deliberately stay in PRESENTING: the shortlist is still valid, so
        # the shopper can check another item, buy a second one, or type
        # "new search" to look at something else entirely.
        return TurnResult(
            reply=reply, stage=state.stage,
            products=[_product_card(r) for r in state.ranked],
            gate=last_gate, gates=gates,
            trajectory=state.trajectory.as_dict(),
            dual_layer=({"consumer_view": last_dual_layer.consumer_view,
                         "enterprise_view": last_dual_layer.enterprise_view} if last_dual_layer else None),
        )

    # Should be unreachable now that PRESENTING never terminates the
    # session, but kept as a safety net rather than silently doing nothing.
    return TurnResult(reply="Something went wrong with this session's state. Start a new one.", stage=state.stage)
