from __future__ import annotations

import re

RED_FLAG_PATTERNS = [
    # FR
    r"\bdouleur\s+thoracique\b",
    r"\boppression\s+thoracique\b",
    r"\b(essoufflement\s+au\s+repos|difficult[ée]\s+respiratoire\s+s[ée]v[èe]re)\b",
    r"\b(perte\s+de\s+connaissance|syncope)\b",
    r"\bparalysie\b",
    r"\bfaiblesse\s+d'un\s+c[ôo]t[ée]\b",
    r"\bh[ée]morragie\b",
    r"\bsang\s+dans\s+les\s+vomissements\b",
    # EN
    r"\b(chest\s+pain|pressure\s+in\s+chest)\b",
    r"\b(severe\s+shortness\s+of\s+breath|can'?t\s+breathe)\b",
    r"\b(fainting|passed\s+out|loss\s+of\s+consciousness)\b",
    r"\b(one\s+sided\s+weakness|face\s+d[ro]op|slurred\s+speech)\b",
    r"\b(uncontrolled\s+bleeding|vomiting\s+blood)\b",
]


def detect_red_flags(text: str) -> list[str]:
    hits: list[str] = []
    norm = text.lower()
    for pat in RED_FLAG_PATTERNS:
        if re.search(pat, norm, flags=re.IGNORECASE):
            hits.append(pat)
    return hits
