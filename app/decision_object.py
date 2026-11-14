"""The Canonical Decision Object.

This is the artefact both halves of the capstone actually share: the
conversational-commerce brief (§6) describes it as "a full decision
trajectory -- stated intent, clarifying exchanges, products retrieved,
ranking rationale, final selection" and calls it "Agentra's input format."
Agentra's own proposal (§5) calls the same thing the Canonical Decision
Object that ADML normalizes everything into.

Building this as one explicit, serializable object (rather than scattering
the same fields across ad-hoc dicts) is what lets this shopping agent and
ASIDE's security gate compose as a pipeline instead of two unrelated demos:
this object is what gets shown on screen, what gets sent to ASIDE for
evaluation, and what would feed a future attribution/dashboard layer without
re-deriving any of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.intent import IntentSlots
from app.retrieval import RankedProduct


@dataclass
class ClarifyingExchange:
    question: str
    answer: str


@dataclass
class DecisionTrajectory:
    session_id: str
    agent_id: str
    user_id: str
    stated_intent: str
    slots: dict
    clarifying_exchange: list[ClarifyingExchange] = field(default_factory=list)
    products_retrieved: list[dict] = field(default_factory=list)
    ranking_rationale: str = ""
    final_selection: dict | None = None
    gate_decision: dict | None = None
    # Dual-layer output (Patent Claim 2) -- one attribution pass, two views.
    # See app/behavioral_profile.py. These always hold the MOST RECENT
    # purchase evaluation (a session can now evaluate several products
    # without ending -- see `purchase_history` for the full list).
    consumer_view: dict | None = None
    enterprise_view: dict | None = None
    # Counterfactual Decision Checker (CDC) result for the CURRENT shortlist
    # (see app/counterfactual.py) -- computed once at retrieval time, not per
    # purchase, since it's a check on the shortlist itself, not on any one
    # item the shopper eventually buys.
    counterfactual: dict | None = None
    # Every purchase evaluated this session, in order -- fixes the earlier
    # "session ends after one product" behavior: a shopper can check as many
    # shortlist items as they want, and this is the full record of that.
    purchase_history: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def start(cls, session_id: str, agent_id: str, user_id: str, stated_intent: str) -> "DecisionTrajectory":
        return cls(session_id=session_id, agent_id=agent_id, user_id=user_id,
                   stated_intent=stated_intent, slots={})

    def record_slots(self, slots: IntentSlots) -> None:
        self.slots = {
            "occasion": slots.occasion,
            "category": slots.category,
            "budget_max": slots.budget_max,
            "wants_durability": slots.wants_durability,
        }

    def record_clarifying(self, question: str, answer: str) -> None:
        self.clarifying_exchange.append(ClarifyingExchange(question=question, answer=answer))

    def record_retrieval(self, ranked: list[RankedProduct], rationale: str) -> None:
        self.products_retrieved = [
            {
                "id": r.product.id,
                "name": r.product.name,
                "price": r.product.price,
                "merchant": r.product.merchant,
                "is_new_merchant": r.product.is_new_merchant,
                "score": r.score,
                "justification": r.justification,
            }
            for r in ranked
        ]
        self.ranking_rationale = rationale

    def record_final_selection(self, ranked: RankedProduct) -> None:
        self.final_selection = {
            "id": ranked.product.id,
            "name": ranked.product.name,
            "price": ranked.product.price,
            "merchant": ranked.product.merchant,
            "is_new_merchant": ranked.product.is_new_merchant,
            "justification": ranked.justification,
        }

    def record_gate(self, gate) -> None:
        self.gate_decision = {
            "decision": gate.decision,
            "risk_score": gate.risk_score,
            "confidence": gate.confidence,
            "reasoning": gate.reasoning,
            "unreachable": gate.unreachable,
        }

    def record_counterfactual(self, counterfactual: dict) -> None:
        self.counterfactual = counterfactual

    def record_dual_layer(self, dual_layer) -> None:
        self.consumer_view = dual_layer.consumer_view
        self.enterprise_view = dual_layer.enterprise_view

    def record_purchase_evaluation(self, product_id: str, product_name: str, gate, dual_layer) -> None:
        """Append one product's evaluation to the session's running history,
        without discarding earlier ones -- this is what lets a shopper check
        several shortlist items in one session instead of the session ending
        after the first. `evaluated_at` lets the Enterprise Dashboard sort
        entries chronologically ACROSS sessions, not just within one."""
        self.purchase_history.append({
            "product_id": product_id,
            "product_name": product_name,
            "decision": gate.decision,
            "risk_score": gate.risk_score,
            "reasoning": gate.reasoning,
            "consumer_view": dual_layer.consumer_view,
            "enterprise_view": dual_layer.enterprise_view,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "stated_intent": self.stated_intent,
            "slots": self.slots,
            "clarifying_exchange": [vars(c) for c in self.clarifying_exchange],
            "products_retrieved": self.products_retrieved,
            "ranking_rationale": self.ranking_rationale,
            "final_selection": self.final_selection,
            "gate_decision": self.gate_decision,
            "consumer_view": self.consumer_view,
            "enterprise_view": self.enterprise_view,
            "counterfactual": self.counterfactual,
            "purchase_history": self.purchase_history,
            "created_at": self.created_at,
        }
