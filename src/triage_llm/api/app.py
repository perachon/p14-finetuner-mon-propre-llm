from __future__ import annotations

from fastapi import FastAPI, HTTPException

from triage_llm.api.audit import AuditStore
from triage_llm.api.model import make_backend_from_env
from triage_llm.eval.safety import detect_red_flags
from triage_llm.schemas import TriageDecision, TriageRequest, TriageResponse

app = FastAPI(title="CHSA - Agent IA Triage (POC)")

audit = AuditStore()
backend = make_backend_from_env()


def _contains_any(message: str, keywords: list[str]) -> bool:
    msg = message.lower()
    return any(k in msg for k in keywords)


def _classify_priority_non_red_flag(message: str, lang: str) -> str:
    """Classifie une priorité *hors red-flags*.

    Heuristique minimale (POC): vise à exploiter les 3 niveaux demandés
    sans prétendre remplacer un protocole de triage clinique.
    """

    msg = message.lower()

    mild_fr = ["rhume", "nez qui coule", "mal de gorge", "toux", "maux de tête", "migraine"]
    mild_en = ["cold", "runny nose", "sore throat", "cough", "headache"]

    concerning_fr = [
        "fièvre",
        "vomis",
        "vomissement",
        "diarrh",
        "déshydrat",
        "douleur",
        "essouff",
        "difficulté à respir",
        "fatigue intense",
    ]
    concerning_en = [
        "fever",
        "vomit",
        "vomiting",
        "diarrh",
        "dehydrat",
        "pain",
        "shortness of breath",
        "breathing difficulty",
        "severe fatigue",
    ]

    mild = mild_fr if lang == "fr" else mild_en
    concerning = concerning_fr if lang == "fr" else concerning_en

    # If nothing concerning is detected and some mild symptoms appear, classify as deferred.
    if _contains_any(msg, mild) and not _contains_any(msg, concerning):
        return "urgence_differee"

    # Default in POC: moderate urgency (clinical review soon).
    return "urgence_moderee"


def _adaptive_follow_up_questions(
    message: str,
    lang: str,
    context: dict,
    *,
    is_red_flag: bool,
) -> list[str]:
    """Construit un questionnaire de suivi *adaptatif* (POC).

    - Questions de base (âge, début, intensité).
    - Ajouts conditionnels selon mots-clés.
    - Filtre les questions déjà couvertes par `context`.
    """

    context = context or {}
    msg = (message or "").lower()

    if lang == "fr":
        base = [
            ("age", "Quel âge avez-vous ?"),
            ("onset", "Depuis quand les symptômes ont-ils commencé ?"),
        ]
        extras: list[tuple[str, str]] = []
        if "fièvre" in msg:
            extras.append(("temperature", "Quelle est la température maximale mesurée ?"))
        if any(k in msg for k in ["vomis", "vomissement", "diarrh", "gastro", "naus"]) :
            extras.append(
                (
                    "hydration",
                    "Pouvez-vous boire et uriner normalement "
                    "(signes de déshydratation) ?",
                )
            )
        if "douleur" in msg:
            extras.append(
                (
                    "pain_scale",
                    "Sur une échelle de 0 à 10, quelle est "
                    "l’intensité de la douleur ?",
                )
            )
        if any(k in msg for k in ["toux", "essouff", "respir"]) :
            extras.append(("breathing", "Avez-vous une gêne respiratoire au repos ou à l’effort ?"))

        if is_red_flag:
            extras.extend(
                [
                    (
                        "location",
                        "Où êtes-vous actuellement "
                        "(domicile, rue, seul/avec quelqu'un) ?",
                    ),
                    ("can_speak", "Pouvez-vous parler en phrases complètes ?"),
                ]
            )

        candidates = base + extras
        out: list[str] = []
        for key, question in candidates:
            if key not in context:
                out.append(question)
        return out[:6]

    # EN
    base = [
        ("age", "What is your age?"),
        ("onset", "When did symptoms start?"),
    ]
    extras = []
    if "fever" in msg:
        extras.append(("temperature", "What is the highest measured temperature?"))
    if any(k in msg for k in ["vomit", "vomiting", "diarrh", "nausea"]) :
        extras.append(("hydration", "Can you drink and urinate normally (any dehydration signs)?"))
    if "pain" in msg:
        extras.append(("pain_scale", "On a scale from 0 to 10, how severe is the pain?"))
    if any(k in msg for k in ["cough", "shortness of breath", "breath", "breathing"]) :
        extras.append(("breathing", "Any breathing difficulty at rest or with activity?"))
    if is_red_flag:
        extras.extend(
            [
                ("location", "Where are you right now (home/outside, alone/with someone)?"),
                ("can_speak", "Can you speak in full sentences?"),
            ]
        )

    candidates = base + extras
    out = []
    for key, question in candidates:
        if key not in context:
            out.append(question)
    return out[:6]


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

        priority = _classify_priority_non_red_flag(req.patient_message, req.lang)
        if req.lang == "fr":
            next_steps = (
                [
                    "Surveillez l’évolution et reconsultez si aggravation.",
                    "Si les symptômes persistent, prenez rendez-vous avec un "
                    "professionnel de santé.",
                ]
                if priority == "urgence_differee"
                else [
                    "Si aggravation ou malaise important, contactez les urgences.",
                    "Sinon, consultez un professionnel de santé rapidement.",
                ]
            )
        else:
            next_steps = (
                [
                    "Monitor symptoms and seek care if they worsen.",
                    "If symptoms persist, book a clinician appointment.",
                ]
                if priority == "urgence_differee"
                else [
                    "If symptoms worsen or you feel very unwell, seek emergency care.",
                    "Otherwise, contact a clinician soon.",
                ]
            )

        decision = TriageDecision(
            priority=priority,
            explanation=raw[:400],
            recommended_next_steps=next_steps,
            red_flags=red_flags,
        )
        follow_up_questions = _adaptive_follow_up_questions(
            req.patient_message,
            req.lang,
            req.context,
            is_red_flag=False,
        )
    else:
        follow_up_questions = _adaptive_follow_up_questions(
            req.patient_message,
            req.lang,
            req.context,
            is_red_flag=True,
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
