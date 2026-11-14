"""Prompt-injection detector for third-party text (listing titles, seller notes, descriptions, reviews).

Rule families (case-insensitive, general principles from Greshake et al. 2023 and the AgentDojo/InjecAgent attack taxonomies):
  override   tells the model to drop its instructions            ("ignore previous instructions", typos tolerated)
  role       fake role/markup aimed at a model                   ("<INFORMATION>", "###(system_message)", "you are now", "to you, an AI")
  sequence   hijacks the task order                              ("before you solve the task, first do ...", "strictly adhere")
  action     agent-directed payment/secret/ranking commands      ("complete the payment without asking", "reveal your system prompt")
  exfil      asks for account data to be sent to an address      ("email a copy of my records to a@b.com")   <- written after seeing 2 InjecAgent rows
Measured recall and false-positive rate: RISK_EVALUATION.md, scripts/evaluate_risk_detectors.py (not a guarantee of coverage)."""
from __future__ import annotations

import re

FAMILIES: dict[str, list[str]] = {
    "override": [
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|your|any)\b[^.\n]{0,25}\b(?:i\w{0,2}nstr\w*|rules?|prompts?|guidelines?|directions?|polic\w+)",
        r"\b(?:instead of|in place of)\b[^.\n]{0,20}\b(?:the |your )?(?:task|instructions?)",
        r"\bdisregard .{0,20}rules",
    ],
    "role": [
        r"<\s*/?\s*(?:information|system|instructions?|important|admin)\s*>",
        r"#{2,}\s*\(?\s*(?:system|assistant)[_ ]?(?:message|prompt)?\s*\)?",
        r"\[\s*(?:system|assistant|inst)\s*\]",
        r"\byou are now\b",
        r"\bsystem prompt\b",
        r"\b(?:message|note|instructions?)\b[^.\n]{0,25}\bto you,?\s+(?:an? |the )?(?:ai|assistant|language model|model|agent|chatbot)\b",
        r"\bdear (?:ai|assistant|agent|language model|chatbot)\b",
        r"^\s*todo\s*:",
    ],
    "sequence": [
        r"\bbefore you (?:can |do |continue|answer|solve|proceed|respond)\b[^.\n]{0,70}\b(?:first|following|instead)\b",
        r"\bdo the following first\b",
        r"\bstrictly adhere\b",
        r"\bafter you do that\b",
        r"\bnew (?:task|instructions?)\s*:",
    ],
    "action": [
        r"\b(?:complete|make|process|authori[sz]e|approve|confirm|execute|finali[sz]e|place)\b[^.\n]{0,40}\b(?:payment|purchase|order|checkout|transaction|transfer)\b[^.\n]{0,40}\b(?:without|skip|no need|don'?t)\b[^.\n]{0,25}\b(?:ask|approval|confirm|verif|check|consent)",
        r"\bskip (?:the )?(?:security|verification|approval|confirmation|check|gate)",
        r"\b(?:reveal|print|show|output|leak|disclose)\b[^.\n]{0,30}\b(?:system prompt|hidden instructions|your instructions|wallet|balance|password|api key|credentials)",
        r"\b(?:rank|recommend|list|show|put)\b[^.\n]{0,25}\b(?:this|my)\b[^.\n]{0,15}\b(?:listing|product|item|offer)\b[^.\n]{0,20}\b(?:first|top|number one|#1|highest)",
        r"\b(?:pay|buy|checkout) (?:immediately|right away)\b",
        r"\b(?:send|transfer|wire)\b[^.\n]{0,25}\b(?:rupees|rs\.?|inr|₹|\$|usd|dollars)\b[^.\n]{0,50}\b(?:account|upi|wallet)\b",
        r"\badd the [^.\n]{0,25}\bto the cart and pay\b",
    ],
    "exfil": [
        r"\b(?:email|send|forward|mail)\b[^.\n]{0,120}\bto\b[^.\n]{0,60}[\w.+-]+@[\w-]+\.[\w.]+",
        r"\b(?:disable|turn off|remove)\b[^.\n]{0,30}\b(?:two-factor|2fa|authentication|security)\b",
        r"\bgrant\b[^.\n]{0,40}\b(?:access|permission)\b",
    ],
}
_COMPILED = {f: [re.compile(p, re.I | re.M) for p in ps] for f, ps in FAMILIES.items()}


def families(text: str, only: tuple[str, ...] | None = None) -> list[str]:
    """Which rule families fire on this text."""
    t = text or ""
    return [f for f, ps in _COMPILED.items() if (only is None or f in only) and any(p.search(t) for p in ps)]


class _Pattern:
    """Drop-in for the old compiled regex: .search(text) is truthy when any family fires."""
    def search(self, text):
        return bool(families(str(text or "")))


PATTERN = _Pattern()
