# Rapport technique — POC Agent IA de triage médical (CHSA)

Projet 14 — Fine-tunez votre propre LLM (OpenClassrooms)

**Contrainte livrable**: PDF, 20 pages max, mission uniquement.

**Disclaimer**: POC éducatif. Ne remplace pas un avis médical.

---

## 1) Résumé exécutif

### Objectif

Démontrer la faisabilité technique d’un système de triage médical assisté par IA, couvrant:

- constitution d’un dataset médical bilingue (FR/EN),
- spécialisation d’un LLM par **SFT (LoRA)** puis **DPO**,
- exposition d’un endpoint d’inférence (API),
- mise en place d’une **CI/CD** et d’un **endpoint cloud** optimisé via **vLLM**.

### Résultats (POC)

- Données prêtes entraînement (JSONL + splits):
  - SFT: **5500** exemples (`data/processed/sft.jsonl`)
  - DPO: **3000** paires (`data/processed/dpo.jsonl`)
- Modèle: `Qwen/Qwen3-1.7B-Base` spécialisé via adapters LoRA (SFT) puis aligné par préférences (DPO).
- Sécurité: détection de **red flags** par règles (court-circuit du modèle) pour prioriser la prudence.
- Déploiement: API FastAPI + audit SQLite; backend Windows-friendly (Transformers+PEFT) et backend cloud (vLLM OpenAI-compatible).
- CI/CD: GitHub Actions (lint+tests) + build Docker.

### Mesures et validations (pré-PDF)

- **Latence**: mesures P50/P95 effectuées sur 10 requêtes (warmup=1) pour le backend local (Transformers+PEFT) et pour les backends `stub` (local + cloud).
- **Qualitatif / sécurité**: validations “red flags” effectuées (court-circuit vers `urgence_maximale`) et journalisées via l’audit.

Procédure de mesure (reproductible, si re-run nécessaire):

- Démarrer l’API localement (Windows): `scripts/run_api.ps1` avec `TRIAGE_BACKEND=transformers` (ou `stub` pour un contrôle), idéalement sur un port libre (ex: `-Port 8001`).
- Lancer le benchmark (10 requêtes mesurées, 1 warmup): `python scripts/benchmark_latency.py --base-url http://127.0.0.1:8001 --n 10 --warmup 1 --print markdown`
- Pour le Space HF (CPU/stub): `python scripts/benchmark_latency.py --base-url https://perachon-p14-space.hf.space --n 10 --warmup 1 --print markdown`

Table de mesure (mesures réalisées):

| Environnement | Backend | Matériel | P50 (s) | P95 (s) | Commentaires |
|---|---|---|---:|---:|---|
| Local | transformers+peft | RTX 4050 6GB (Windows) | 13.491 | 13.693 | Mesure réelle sur 10 requêtes (warmup=1) |
| Local | stub (FastAPI) | CPU (Windows) | 0.020 | 0.042 | Mesure réelle sur 10 requêtes; baseline API sans modèle |
| Cloud | stub (FastAPI) | CPU (HF Spaces) | 0.227 | 0.306 | Mesure réelle sur 10 requêtes; réseau inclus |
| Cloud | vLLM | GPU (Linux) | — | — | Non déployé (pas de GPU gratuit). Procédure ci-dessous |

### Recommandation

- **Go** pour un POC supervisé (shadow mode) avec périmètre maîtrisé, audit, et règles de sécurité.
- **No-go** pour usage autonome sans validation clinique, gouvernance, monitoring, et garde-fous renforcés.

---

## 2) Contexte, besoins et exigences

### Besoins fonctionnels

- Entrée: message patient (FR/EN)
- Sortie: priorité de triage (urgence maximale / modérée / différée), explication prudente, questions de suivi, étapes recommandées
- Traçabilité: journaliser les interactions

### Contraintes (sécurité / conformité)

- Le POC ne doit pas fournir de diagnostic “certain” ni de posologie.
- Les cas graves doivent être détectés et orientés vers une évaluation urgente.
- RGPD: pas de données SIH réelles; anonymisation disponible si nécessaire.

---

## 3) Données (Étape 1)

### 3.1 Source principale, licence et ingestion

Source principale (Hugging Face Datasets):

- Dataset: `cyrille-elie/CHSA-Triage-Medic-Full-Dataset`
- Licence: MIT (tag HF `license:mit`)
- Conversion:
  - config `sft_medical_dataset` → JSONL SFT
  - config `dpo_dataset` → JSONL DPO

Les détails (source, transformations, publication d’artefacts dérivés) sont consignés dans `data/SOURCES.md`.

### 3.2 Schémas et formats

- SFT: `SFTRecord` (instruction/input/output/lang + métadonnées)
- DPO: `DPORecord` (prompt/chosen/rejected/lang + métadonnées)

Fichiers générés (existant dans le repo):

- `data/processed/sft.jsonl` (**5500** lignes)
- `data/processed/dpo.jsonl` (**3000** lignes)
- splits train/val/test (seed 42):
  - SFT train/val/test = 4950 / 275 / 275
  - DPO train/val/test = 2700 / 150 / 150
- journal des transformations: `data/processed/audit_log.jsonl`

### 3.3 RGPD et anonymisation

- Périmètre: pas de données SIH réelles.
- Pipeline d’anonymisation disponible: Presidio (replace/mask/redact) si une source contenait des données personnelles.
- Traçabilité: audit log + revue par échantillonnage.

Document de justification: `data/RGPD_PROCESS.md`.

---

## 4) Modèle & entraînement (Étape 2)

### 4.1 Modèle de base

- Base: `Qwen/Qwen3-1.7B-Base`
- Raison du choix: compromis taille/qualité adapté au prototypage et au fine-tuning par adapters LoRA.

### 4.2 SFT + LoRA

Objectif: instruction-tuning pour obtenir des réponses plus:

- prudentes,
- structurées,
- orientées triage (questions + étapes suivantes),
- bilingues (FR/EN).

Configuration (script `scripts/train_sft_lora.py`, defaults):

- `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
- `learning_rate=2e-4`
- `max_seq_length=1024`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`

Runs “long” low-VRAM (script `scripts/run_long_experiments.ps1`):

- `max_seq_length=128`, `max_steps=800`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`
- LoRA: `r=4`, `alpha=8`, `target_modules=[q_proj, v_proj]`
- `fp16` activé (GPU)

### 4.3 DPO

Objectif: alignement par préférences (favoriser les réponses “choisies” vs “rejetées”).

Configuration (script `scripts/train_dpo.py`, defaults):

- `beta=0.1`
- `max_length=256`
- `learning_rate=1e-5`
- `per_device_train_batch_size=1`

Runs “long” (script `scripts/run_long_experiments.ps1`):

- `max_length=128`, `max_steps=400`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`

### 4.4 Artefacts et publication

- Checkpoints locaux: `checkpoints/…`
- Adapters LoRA publiés (SFT & DPO, runs courts et longs): `perachon/p14-model`

---

## 5) Évaluation & sécurité

### 5.1 Méthodologie (POC)

- Validation d’ingénierie: schémas, tests unitaires, CI.
- Validation qualitative: batterie de prompts (FR/EN) + analyse des cas.

### 5.2 Garde-fous: red flags

En présence de red flags (ex: douleur thoracique, syncope, hémorragie), la logique de sécurité:

- force une priorité **urgence_maximale**,
- fournit des recommandations prudentes,
- n’appelle pas le modèle.

Limites: règles incomplètes → faux négatifs/positifs possibles. Recommandation: enrichir la couverture, gérer les négations, et valider sur un jeu clinique.

### 5.3 Batterie de démonstration (soutenance)

Prompts suggérés:

- Cas non red-flag (FR): “mal de gorge 2 jours, nez qui coule, pas de fièvre, hydratation OK”
- Cas non red-flag (EN): “sore throat, runny nose, no fever, can eat and drink”
- Cas modéré: “diarrhée 24h, pas de sang, je bois, pas d’étourdissement”
- Cas red-flag: “douleur thoracique + essoufflement depuis 30 minutes”

Éléments consignés dans ce rapport:

- validations API (triage + audit) via exemples réels (section 6.3),
- mesures de latence P50/P95 (section 1),
- cas “red flags” (section 6.3) pour vérifier le court-circuit sécurité.

---

## 6) Déploiement & validation (Étape 3)

### 6.1 API

Endpoints:

- `GET /health`
- `POST /triage`
- `GET /audit/{interaction_id}`

Traçabilité:

- stockage SQLite: `runs/audit/audit.db`
- requêtes/réponses enregistrées en JSON (UTF-8)

### 6.2 Backends modèle (sélection par variable d’environnement)

- `TRIAGE_BACKEND=stub`: CI (aucun modèle)
- `TRIAGE_BACKEND=transformers`: Transformers + PEFT (Windows-friendly)
- `TRIAGE_BACKEND=vllm-openai`: forward vers vLLM (`/v1/chat/completions`)

### 6.3 Endpoint cloud vLLM

- vLLM est conçu pour tourner sur Linux + GPU.
- Un dossier de déploiement est fourni pour Hugging Face Spaces Docker + GPU.

Guide: `cloud/hf_spaces_vllm/README.md`.

Endpoint cloud de démonstration (gratuit, CPU, sans vLLM):

- Space Hugging Face: https://huggingface.co/spaces/perachon/p14-space
- URL publique API: https://perachon-p14-space.hf.space
- Swagger: https://perachon-p14-space.hf.space/docs

Validation (extrait réel, appel `POST /triage` via Swagger):

- `interaction_id`: `397e07b5-9e01-48cb-a95d-a7f94acb980b`
- `priority`: `urgence_moderee`
- `backend`: `stub` (réponse de démonstration, sans modèle)
- Audit associé: https://perachon-p14-space.hf.space/audit/397e07b5-9e01-48cb-a95d-a7f94acb980b

Validation red-flag (extrait réel: saignement important → court-circuit sécurité):

- `interaction_id`: `900ddad7-be9e-4ce3-bed7-7dea5a01e32d`
- `priority`: `urgence_maximale`
- `red_flags` (lisible): `saignement abondant`
- `red_flags` (regex): `\\bsaigne?\\s+(beaucoup|[ée]norm[ée]ment|abondamment)\\b`
- Audit associé: https://perachon-p14-space.hf.space/audit/900ddad7-be9e-4ce3-bed7-7dea5a01e32d

Endpoints cloud (état actuel):

- Endpoint cloud **gratuit** (Swagger, backend `stub`): https://perachon-p14-space.hf.space/docs
- Endpoint **vLLM GPU** (OpenAI-compatible): non déployé (pas de GPU gratuit)
- GPU / provider: non déployé (voir option recommandée ci-dessous)

Note (choix “gratuit”):

- Les offres gratuites ne fournissent généralement pas de **GPU NVIDIA**. Or vLLM est conçu pour l’inférence GPU (optimisation et latence). En conséquence, un endpoint vLLM “rapide” n’a pas pu être maintenu sur une infra gratuite.
- Pour rester conforme à la démarche demandée (solution efficace identifiée), le POC inclut:
  - un backend **`vllm-openai`** côté API (forward vers `/v1/chat/completions`),
  - un packaging Docker prêt à déployer sur une infra GPU Linux.

Remarque: l’endpoint cloud gratuit ci-dessus sert à la **démo API** (contrat, audit, Swagger) et utilise `TRIAGE_BACKEND=stub`.

Option recommandée si budget disponible (déploiement en < 1h):

- Provider: Hugging Face Spaces (Docker + GPU)
- GPU minimal conseillé: Nvidia T4 (≈15 GB)
- Procédure: suivre `cloud/hf_spaces_vllm/README.md`, récupérer l’URL publique, mesurer la latence, puis arrêter l’instance.

---

## 7) CI/CD

GitHub Actions:

- lint: `ruff check .`
- tests: `pytest -q`
- build Docker: `docker/Dockerfile`

Objectif: éviter les régressions sur schémas de données et endpoints API.

---

## 8) Recommandations et passage à l’échelle

### Court terme

- Formaliser un jeu d’évaluation (50–200 cas) revu par un clinicien.
- Mesurer latence/débit et taux de sorties à risque.
- Structurer la sortie (JSON strict) et valider côté serveur.

### Moyen terme

- Monitoring (dérive de prompts, alertes, revue clinique, traçabilité).
- Gouvernance (process de versioning datasets/modèles, conformité, DPIA si données réelles).

---

## Annexes

### A) Artefacts Hugging Face

- Modèle (adapters LoRA): `perachon/p14-model`
- Dataset du projet (privé): `perachon/p14-dataset`

### B) Commandes (rappel)

- Runs longs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_long_experiments.ps1 -SftMaxSteps 800 -DpoMaxSteps 400 -SeqLen 128
```

---

## Export PDF (Windows)

Option simple: extension VS Code “Markdown PDF”.

- Ouvrir ce fichier
- Commande VS Code: “Markdown PDF: Export (pdf)”

(Alternative: pandoc si installé.)
