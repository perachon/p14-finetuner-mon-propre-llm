# Sources de données (à compléter)

Documentez ici, pour chaque source utilisée :

- Nom du dataset (ex: MediQA, FrenchMedMCQA, MedQuAD, UltraMedical-Preference)
- URL / référence
- Licence
- Mode d'acquisition (HF / téléchargement / accès institutionnel)
- Transformations appliquées (nettoyage, filtrage, traduction éventuelle)
- Justification RGPD / anonymisation

## Source principale (HF)

- Nom : `cyrille-elie/CHSA-Triage-Medic-Full-Dataset`
- URL : https://huggingface.co/datasets/cyrille-elie/CHSA-Triage-Medic-Full-Dataset
- Licence : MIT (tag HF `license:mit`)
- Mode d'acquisition : Hugging Face Datasets (`datasets.load_dataset`)
- Sous-configs utilisées :
	- `sft_medical_dataset` (instruction, response, source_dataset, language) → converti en JSONL SFT du repo
	- `dpo_dataset` (prompt, chosen, rejected) → converti en JSONL DPO du repo
- Transformations appliquées :
	- Conversion de champs : `response` → `output`, `language` → `lang`, ajout de `id/source/created_at`
	- Fusion des splits HF puis re-split `train/val/test` par le builder du repo (seed 42)
	- Pas de traduction
- RGPD / anonymisation :
	- Source publique (pas de données patients attendues). Le pipeline d'anonymisation Presidio est disponible si nécessaire.
	- Flag `pii_redacted=true` mis par défaut lors de l'ingestion (hypothèse "pas de PII"), à confirmer selon le besoin projet.

## Artifacts dérivés publiés (HF Model)

- Dépôt modèle : `perachon/p14-model`
- Contenu : adapters LoRA (PEFT) entraînés sur la source ci-dessus
	- Runs “courts” : `adapters/sft/`, `adapters/dpo/`
	- Runs “long” : `adapters/sft_long_20260318_1657/`, `adapters/dpo_long_20260318_1657/`
