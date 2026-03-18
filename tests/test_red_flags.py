from __future__ import annotations

from triage_llm.eval.safety import detect_red_flags


def test_red_flags_detect_heavy_bleeding_and_fainting_fr() -> None:
    msg = "je me suis ouvert la main, je saigne énormément et j'ai envie de m'évanouir"
    hits = detect_red_flags(msg)
    assert hits, "Expected red flags to be detected"


def test_red_flags_detect_bleeding_en() -> None:
    msg = "I cut my hand and I'm bleeding heavily, I feel faint"
    hits = detect_red_flags(msg)
    assert hits, "Expected red flags to be detected"
