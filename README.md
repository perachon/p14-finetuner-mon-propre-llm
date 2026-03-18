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

## Docs (Étapes 0/1)

- Étape 0 (théorie SFT/DPO) : `docs/00_theorie_sft_dpo.md`
- Étape 1 (données, RGPD, pipeline) : `docs/01_etape1_donnees.md`

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

### Windows + GPU (CUDA) — recommandé pour l'entraînement

Sur Windows, l'entraînement GPU nécessite une installation **CUDA** de PyTorch (la version PyPI standard est souvent CPU-only).

Notes pratiques :

- Les wheels CUDA de PyTorch sont très volumineuses (plusieurs Go) : assurez-vous d'avoir assez d'espace disque.
- Si votre disque `C:` est presque plein, redirigez `TEMP/TMP` vers un disque avec de la place (ex: `D:`) pendant l'installation.

Exemple PowerShell (Python 3.12 conseillé) :

```powershell
# (Option) installer un Python 3.12 via uv (sans admin)
uv python install 3.12

# venv dédiée
uv venv --python 3.12 .venv312
\.venv312\Scripts\python.exe -m ensurepip --upgrade

# éviter les erreurs "No space left on device" si C: est plein
$env:TEMP = 'D:\pip-tmp'
$env:TMP = 'D:\pip-tmp'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null

# PyTorch CUDA (adapter cu124 si besoin)
\.venv312\Scripts\python.exe -m pip install -U pip setuptools wheel
\.venv312\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# deps projet
\.venv312\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt

# check GPU
\.venv312\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Limites Windows :

- `bitsandbytes` et `vllm` sont désactivés par défaut sur Windows dans `requirements.txt`.
- Le serving vLLM (Étape 3) est donc plutôt à faire sur un environnement Linux/Cloud, même si vous développez depuis Windows.

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

### Auth Hugging Face (optionnel mais conseillé)

Si vous téléchargez souvent depuis le Hub ou si vous publiez un dataset/modèle, configurez un token Hugging Face.

Deux options :

1) Login global (recommandé)

```bash
huggingface-cli login
```

2) Fichier local `.env` (pratique en projet, ignoré par git)

- Copier `.env.example` → `.env`
- Mettre `HF_TOKEN=...`

Les scripts `scripts/train_sft_lora.py` et `scripts/train_dpo.py` chargent automatiquement `.env` si présent.

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

### Runs “long” (comparaison)

Un script reproductible est fourni pour lancer **SFT long puis DPO long** avec des dossiers séparés (pratique pour comparer avec les runs courts) :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_long_experiments.ps1 -SftMaxSteps 800 -DpoMaxSteps 400 -SeqLen 128
```

Sorties (exemple) :

- `checkpoints/qwen3-1.7b-sft-lora_LONG_20260318_1657/`
- `checkpoints/qwen3-1.7b-dpo_LONG_20260318_1657/`

## Artifacts Hugging Face

- Modèle (adapters LoRA) : `perachon/p14-model`
  - `adapters/sft/` et `adapters/dpo/` (runs “courts”)
  - `adapters/sft_long_20260318_1657/` et `adapters/dpo_long_20260318_1657/` (runs “long”)
- Dataset (privé) : `perachon/p14-dataset`

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
