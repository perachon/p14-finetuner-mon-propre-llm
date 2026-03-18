# Rapport technique — POC Agent IA de triage médical (CHSA)

Contexte: Projet 14 — Fine-tunez votre propre LLM (OpenClassrooms)

Contraintes de la mission:

- Format: **PDF**, **20 pages maximum** (mission uniquement)
- Pipeline: **données → SFT (LoRA) → DPO → API → CI/CD → endpoint cloud (vLLM)**

Disclaimer: ce POC est éducatif et ne constitue pas un dispositif médical. Toute utilisation réelle doit se faire sous supervision clinique.

---

## 1) Résumé exécutif (≈ 1 page)

### Problème

Les services d’urgences font face à une surcharge et à une hétérogénéité des demandes. L’objectif du POC est de démontrer la faisabilité technique d’un **agent intelligent** capable de:

- recueillir un message patient,
- détecter des **signes de gravité (red flags)**,
- proposer une **priorité de triage** (urgence maximale / modérée / différée),
- poser des **questions de suivi**,
- tracer l’interaction (audit),
- et être déployable sous forme d’endpoint.

### Ce qui a été réalisé

1) Données: création d’un dataset médical **bilingue FR/EN** au format **JSONL** et export possible **Hugging Face Datasets**, avec schémas validés.

2) Modèle: spécialisation de `Qwen/Qwen3-1.7B-Base` via:

- **SFT + LoRA** (instruction-tuning)
- puis **DPO** (alignement par préférences)

3) Déploiement:

- API FastAPI avec endpoints `GET /health`, `POST /triage`, `GET /audit/{id}`
- backend Windows-friendly (Transformers + PEFT) pour exécuter le POC sur machine locale
- backend **vLLM OpenAI-compatible** pour satisfaire l’exigence d’un endpoint cloud optimisé par vLLM

4) CI/CD: pipeline GitHub Actions avec lint/tests et build Docker.

### Résultats clés (à compléter avec mesures)

- Qualité (qualitative): le modèle répond de manière plus structurée (prudence, questions, étapes suivantes) après SFT/DPO.
- Sécurité: détection de red flags par règles (court-circuit du modèle) afin de privilégier la prudence.
- Latence: **TODO** mesurer P50/P95 en local (Transformers+PEFT) et en cloud (vLLM) sur un set de 10 prompts.

### Recommandation (go/no-go)

- Go pour un **POC clinique supervisé** (shadow mode) avec périmètre maîtrisé, journalisation, et règles de sécurité.
- No-go pour une utilisation autonome: nécessite davantage de données, une évaluation clinique plus robuste, et des garde-fous supplémentaires (policies, monitoring, red teaming).

---

## 2) Contexte & exigences

### Exigences fonctionnelles

- Entrée: message patient (FR/EN)
- Sortie attendue: priorité de triage + explication + questions de suivi + recommandations prudentes
- Traçabilité: conservation des requêtes/réponses pour audit

### Exigences non fonctionnelles

- Sécurité: détecter les cas graves et recommander une évaluation urgente
- RGPD: pas de données SIH réelles, anonymisation possible via pipeline
- Reproductibilité: scripts, seeds, artefacts publiés (adapters), et CI

---

## 3) Données (Étape 1)

### 3.1 Source(s) et licences

Source principale (Hugging Face, licence MIT):

- Dataset: `cyrille-elie/CHSA-Triage-Medic-Full-Dataset`
- Utilisation:
	- config `sft_medical_dataset` → conversion en JSONL SFT
	- config `dpo_dataset` → conversion en JSONL DPO

Références et transformations sont documentées dans `data/SOURCES.md`.

### 3.2 Formats et schémas

Deux formats sont utilisés, avec validation Pydantic:

- SFT: `SFTRecord` (instruction/input/output/lang + métadonnées)
- DPO: `DPORecord` (prompt/chosen/rejected/lang + métadonnées)

Sorties du builder:

- `data/processed/sft.jsonl`, `data/processed/dpo.jsonl`
- `data/processed/splits/` (train/val/test)
- `data/processed/audit_log.jsonl` (journal append-only)
- `data/processed/hf/` (export `save_to_disk`, optionnel)

### 3.3 Séparation entraînement / évaluation

Le builder est conçu pour conserver des jeux d’évaluation clinique séparés (ex: `data/eval/`) afin d’éviter la contamination des splits.

### 3.4 RGPD & anonymisation

Périmètre:

- Le POC n’intègre pas de données patients issues d’un SIH réel.
- Le pipeline d’anonymisation (Presidio) est disponible si une source contenait des informations personnelles.

Mesures:

- stratégies `replace|mask|redact`
- revue manuelle sur échantillon
- traçabilité via `audit_log.jsonl`

Process documenté dans `data/RGPD_PROCESS.md`.

---

## 4) Modèle & entraînement (Étape 2)

### 4.1 Modèle de base

Base: `Qwen/Qwen3-1.7B-Base`

Justification (POC):

- taille raisonnable pour prototypage
- bon compromis pour fine-tuning avec ressources GPU limitées
- support chat template côté tokenizer

### 4.2 SFT + LoRA (instruction-tuning)

Objectif: adapter le style de réponse (prudence, structure, questions de suivi) et le vocabulaire au contexte triage.

Implémentation:

- Entraînement via script `scripts/train_sft_lora.py`
- LoRA configurable (rang, alpha, dropout, modules cibles)

Hyperparamètres par défaut (script):

- `max_seq_length=1024`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-4`
- `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`

Runs “long” low-VRAM (utilisés pour comparaison):

- `max_seq_length=128`
- `max_steps=800`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`
- `lora_r=4`, `lora_alpha=8`
- `target_modules = [q_proj, v_proj]`
- `fp16` activé (GPU)

Artefacts:

- checkpoints locaux dans `checkpoints/`
- publication des adapters LoRA sur Hugging Face: `perachon/p14-model`

### 4.3 DPO (alignement par préférences)

Objectif: favoriser des réponses jugées préférables (prudence, étapes suivantes, réduction des conseils risqués) à partir de paires `chosen` / `rejected`.

Implémentation:

- Entraînement via `scripts/train_dpo.py`
- entrée: JSONL DPO (prompt/chosen/rejected)

Hyperparamètres par défaut (script):

- `beta=0.1`
- `max_length=256`
- `learning_rate=1e-5`
- `per_device_train_batch_size=1`

Runs “long” low-VRAM:

- `max_length=128`
- `max_steps=400`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`

### 4.4 Reproductibilité et traçabilité

- splits reproductibles (seed 42)
- scripts dédiés et paramètres en ligne de commande
- artefacts de modèle publiés (adapters LoRA) pour permettre la réutilisation sans repartir de zéro

---

## 5) Évaluation & sécurité

### 5.1 Approche d’évaluation (POC)

Le POC combine:

- une validation “ingénierie” (tests unitaires, CI, schémas)
- une évaluation qualitative contrôlée (prompts représentatifs)
- des garde-fous de sécurité (red flags)

Tests existants:

- validation des schémas sur `data/sample/*.jsonl`
- endpoint `GET /health`

### 5.2 Garde-fous: red flags

Les symptômes graves sont détectés via expressions régulières (ex: douleur thoracique, essoufflement sévère, syncope, hémorragie). En cas de red flags:

- la priorité est forcée à **urgence_maximale**
- le modèle n’est pas interrogé (principe “safety first”)

Limite: approche rule-based → risque de faux négatifs (couverture incomplète) et faux positifs (sens/ironie). Recommandation: passer à un classifieur dédié (ou une policy LLM + tests) en phase 2.

### 5.3 Batterie de prompts (proposée pour la soutenance)

Objectif: démontrer que le modèle est effectivement appelé lorsque le cas n’est pas un red-flag.

| Catégorie | Exemple (FR) | Attendu |
|---|---|---|
| Bénin | “mal de gorge, nez qui coule, pas de fièvre” | questions + conseils prudents |
| Modéré | “douleur abdominale + diarrhée 24h, hydratation OK” | recommandations + surveillance |
| Red-flag | “douleur thoracique + essoufflement” | urgence_maximale (règles) |

**TODO (rapide)**: consigner 5 à 10 sorties (stub vs transformers-peft vs vLLM) et relever les temps de réponse.

### 5.4 Risques résiduels

- hallucinations (diagnostic inventé)
- conseils inadaptés (médicaments, posologies)
- biais de données (style et couverture symptomatologique)

Mitigation recommandée:

- limiter le scope (triage + questions, pas de diagnostic)
- injecter des disclaimers systématiques
- monitoring, red teaming, et revue clinique

---

## 6) Déploiement & validation (Étape 3)

### 6.1 Architecture

Composants:

- API FastAPI: endpoints de triage + audit
- stockage audit: SQLite (`runs/audit/audit.db`)
- backend modèle (au choix via variable d’environnement)

Backends supportés:

- `stub`: pour CI (aucun modèle chargé)
- `transformers`: Transformers + PEFT (Windows-friendly)
- `vllm-openai`: appel d’un serveur vLLM OpenAI-compatible (cloud)

### 6.2 Endpoint local (Windows)

Script recommandé:

- `scripts/run_api.ps1`

Points d’attention Windows:

- vLLM non disponible → utiliser Transformers+PEFT
- PowerShell 5.1: envoyer le JSON en UTF-8 (corrigé dans `scripts/demo_api.ps1`)

### 6.3 Endpoint cloud (vLLM)

Objectif mission: endpoint déployé sur le cloud optimisé grâce à vLLM.

Solution fournie:

- déploiement sur Hugging Face Spaces (Docker + GPU)
- démarrage vLLM + FastAPI dans le même container
- l’API forward les générations vers vLLM via le backend `vllm-openai`

Guide: `cloud/hf_spaces_vllm/README.md`.

### 6.4 CI/CD (GitHub Actions)

Pipeline:

- `ruff check .`
- `pytest -q` (avec `PYTHONPATH=src`)
- build Docker (`docker/Dockerfile`)

Objectif: garantir qu’une modification ne casse pas les schémas de données et que l’API démarre.

---

## 7) Recommandations stratégiques & passage à l’échelle

### 7.1 Améliorations prioritaires (1 à 4 semaines)

1) Évaluation structurée:

- constituer un set d’éval “triage” (50–200 cas) validé par un clinicien
- mesurer latence, cohérence, taux de recommandations dangereuses

2) Sécurité:

- étendre les red flags (sémantique, négations)
- ajouter une policy de refus (médicaments, dosages)

3) UX/API:

- sortie plus structurée (JSON strict) + validation côté serveur
- séparation claire triage vs recommandations

### 7.2 Industrialisation (1 à 3 mois)

- MLOps: versioning datasets, tracking expériences (MLflow/W&B), registry
- Monitoring: dérive de prompts, alertes, échantillonnage pour revue clinique
- Sécurité: pentest prompt-injection, filtres PII, red teaming

### 7.3 Gouvernance et conformité

- DPIA si données réelles
- validation clinique, procédures de mise à jour
- gestion des accès et des logs (durées de conservation)

---

## Annexes

### A) Artefacts Hugging Face

- Modèle (adapters LoRA): `perachon/p14-model`
- Dataset (privé): `perachon/p14-dataset`

### B) Commandes utiles (rappel)

Build dataset:

```bash
python -m scripts.build_datasets --input_dir data/sample --out_dir data/processed --seed 42
```

SFT LoRA:

```bash
python -m scripts.train_sft_lora --model_name_or_path Qwen/Qwen3-1.7B-Base --sft_jsonl data/processed/splits/sft_train.jsonl --output_dir checkpoints/qwen3-1.7b-sft-lora
```

DPO:

```bash
python -m scripts.train_dpo --model_name_or_path checkpoints/qwen3-1.7b-sft-lora --dpo_jsonl data/processed/splits/dpo_train.jsonl --output_dir checkpoints/qwen3-1.7b-dpo
```

Runs “long”:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_long_experiments.ps1 -SftMaxSteps 800 -DpoMaxSteps 400 -SeqLen 128
```
