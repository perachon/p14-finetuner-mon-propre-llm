from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass
class AnonymizationResult:
    text: str
    entities: list[dict]


class PresidioAnonymizer:
    def __init__(
        self,
        language: str = "fr",
        operators: dict[str, OperatorConfig] | None = None,
    ) -> None:
        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.operators = operators or {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})
        }

    def anonymize(self, text: str, entities: list[str] | None = None) -> AnonymizationResult:
        entities = entities or ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        results = self.analyzer.analyze(text=text, entities=entities, language=self.language)
        anonymized = self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self.operators
        )
        return AnonymizationResult(
            text=anonymized.text,
            entities=[
                {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
                for r in results
            ],
        )


def anonymize_texts(
    texts: Iterable[str],
    language: str = "fr",
    operator: str = "replace",
    new_value: str = "<REDACTED>",
) -> list[AnonymizationResult]:
    engine = PresidioAnonymizer(
        language=language,
        operators={"DEFAULT": OperatorConfig(operator, {"new_value": new_value})},
    )
    return [engine.anonymize(t) for t in texts]
