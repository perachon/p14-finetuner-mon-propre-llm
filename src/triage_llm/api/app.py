from __future__ import annotations

from fastapi import FastAPI, HTTPException

from triage_llm.api.audit import AuditStore
from triage_llm.api.model import make_backend_from_env
from triage_llm.eval.safety import detect_red_flags
from triage_llm.schemas import TriageDecision, TriageRequest, TriageResponse

app = FastAPI(title="CHSA - Agent IA Triage (POC)")

audit = AuditStore()
backend = make_backend_from_env()


def _rule_based_decision(message: str, lang: str) -> tuple[TriageDecision | None, list[str]]:
    red_flags = detect_red_flags(message)
    if red_flags:
        if lang == "fr":
            decision = TriageDecision(
                priority="urgence_maximale",
                explanation=(
                    "Des signes potentiellement graves ont été détectés dans votre message. "
                    "Par prudence, une évaluation médicale urgente est recommandée."
                ),
                recommended_next_steps=[
                    (
                        "Appelez immédiatement les services d'urgence (15/112) "
                        "si vous êtes en danger immédiat."
                    ),
                    "Ne conduisez pas vous-même si vous vous sentez mal.",
                ],
                red_flags=red_flags,
            )
        else:
            decision = TriageDecision(
                priority="urgence_maximale",
                explanation=(
                    "Potential red-flag symptoms were detected. For safety, "
                    "urgent medical assessment is recommended."
                ),
                recommended_next_steps=[
                    "Call emergency services immediately if this is an emergency.",
                    "Do not drive yourself if you feel unwell.",
                ],
                red_flags=red_flags,
            )
        return decision, red_flags
    return None, []


@app.get("/health")
def health() -> dict:
    info = backend.info()
    return {"status": "ok", "backend": info.name, "details": info.details}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    decision, red_flags = _rule_based_decision(req.patient_message, req.lang)
    follow_up_questions: list[str]

    if decision is None:
        # Keep the API contract stable: we use model text as explanation.
        raw = backend.generate(req.patient_message)
        decision = TriageDecision(
            priority="urgence_moderee",
            explanation=raw[:400],
            recommended_next_steps=(
                [
                    "Si aggravation ou malaise important, contactez les urgences.",
                    "Sinon, consultez un professionnel de santé rapidement.",
                ]
                if req.lang == "fr"
                else [
                    "If symptoms worsen or you feel very unwell, seek emergency care.",
                    "Otherwise, contact a clinician soon.",
                ]
            ),
            red_flags=red_flags,
        )
        follow_up_questions = (
            [
                "Quel âge avez-vous ?",
                "Depuis quand les symptômes ont-ils commencé ?",
                "Avez-vous de la fièvre, des vomissements, ou une douleur thoracique ?",
            ]
            if req.lang == "fr"
            else [
                "What is your age?",
                "When did symptoms start?",
                "Do you have fever, vomiting, or chest pain?",
            ]
        )
    else:
        follow_up_questions = (
            [
                "Où êtes-vous actuellement (domicile, rue, seul/avec quelqu'un) ?",
                "Pouvez-vous parler en phrases complètes ?",
                "Avez-vous des antécédents médicaux importants (cœur, poumons) ?",
            ]
            if req.lang == "fr"
            else [
                "Where are you right now (home, outside, alone/with someone)?",
                "Can you speak in full sentences?",
                "Any major medical history (heart/lungs)?",
            ]
        )

    response = TriageResponse(
        interaction_id="",
        decision=decision,
        follow_up_questions=follow_up_questions,
        model_info=backend.info().details,
    )
    interaction_id = audit.save(req.model_dump(mode="json"), response.model_dump(mode="json"))
    response.interaction_id = interaction_id
    return response


@app.get("/audit/{interaction_id}")
def get_audit(interaction_id: str) -> dict:
    row = audit.get(interaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    return row
