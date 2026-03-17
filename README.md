# Projet 14 — Fine-tunez votre propre LLM (POC triage médical)

Ce dépôt contient un **POC** d'agent IA de triage médical (CHSA) couvrant :

- **Étape 1 — Données** : constitution d'un dataset médical **bilingue (FR/EN)**, anonymisation RGPD, formats **JSONL** et **Hugging Face Datasets**.
- **Étape 2 — Modèle** : fine-tuning supervisé (**SFT**) du modèle **Qwen3-1.7B-Base** avec **LoRA**, puis alignement par préférences (**DPO**).
- **Étape 3 — Déploiement** : API **FastAPI** adossée à **vLLM**, conteneurisation **Docker**, et **CI/CD GitHub Actions**.

> Important : ce POC n'est pas un dispositif médical. Les sorties doivent être utilisées en **aide au triage** avec **supervision clinique**.

## Structure

- `src/triage_llm/` : package principal (schémas, données, entraînement, évaluation, API).
- `scripts/` : commandes CLI simples (préparation dataset, entraînement, export).
- `data/sample/` : petits exemples anonymisés pour tests locaux/CI.
- `docker/` : Dockerfile et scripts de lancement.
- `.github/workflows/` : pipeline CI.
- `report/` : gabarit de rapport (à exporter en PDF, ≤ 20 pages).

## Installation (dev)

1) Créer un environnement Python (3.10+ recommandé)

```bash
python -m venv .venv
```

2) Installer les dépendances

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3) (Optionnel) Installer spaCy FR pour améliorer l'anonymisation Presidio

```bash
python -m spacy download fr_core_news_md
```

## Étape 1 — Construire les datasets SFT/DPO

À partir de fichiers locaux (JSON/JSONL/CSV) ou de datasets Hugging Face (si disponibles), vous pouvez :

```bash
python -m scripts.build_datasets \
  --input_dir data/sample \
  --out_dir data/processed \
  --seed 42
```

Sorties attendues :

- `data/processed/sft.jsonl` (~5k paires à terme)
- `data/processed/dpo.jsonl`
- `data/processed/metadata_schema.json`
- `data/processed/splits/` (train/val/test)

## Étape 2 — Entraîner (SFT LoRA puis DPO)

SFT (LoRA) :

```bash
python -m scripts.train_sft_lora \
  --model_name_or_path Qwen/Qwen3-1.7B-Base \
  --sft_jsonl data/processed/splits/sft_train.jsonl \
  --output_dir checkpoints/qwen3-1.7b-sft-lora
```

DPO :

```bash
python -m scripts.train_dpo \
  --model_name_or_path checkpoints/qwen3-1.7b-sft-lora \
  --dpo_jsonl data/processed/splits/dpo_train.jsonl \
  --output_dir checkpoints/qwen3-1.7b-dpo
```

## Étape 3 — API (FastAPI + vLLM)

Lancement local (CPU possible pour dev, GPU recommandé pour vLLM) :

```bash
set MODEL_NAME_OR_PATH=checkpoints/qwen3-1.7b-dpo
uvicorn triage_llm.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints :

- `GET /health`
- `POST /triage` (questionnaire + triage)
- `GET /audit/{interaction_id}`

## Rapport

Le gabarit est dans `report/rapport_template.md`.

## Licence & données

Les jeux de données médicaux ont des licences variées. Documentez l'origine et la licence dans `data/SOURCES.md`.
