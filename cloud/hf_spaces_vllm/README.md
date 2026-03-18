# Endpoint cloud (vLLM) — guide de déploiement

Objectif: satisfaire le livrable **"Endpoint de démonstration déployé sur le cloud"** avec une inférence rapide via **vLLM**.

Contrainte: le projet est développé sous Windows, mais **vLLM ne tourne pas sur Windows**. Le déploiement cloud se fait donc sur une machine **Linux + GPU NVIDIA**.

## Option recommandée (simple) : Hugging Face Spaces (Docker + GPU)

Hugging Face Spaces est un fournisseur cloud acceptable pour un POC: vous obtenez une URL publique, et vous pouvez choisir une machine GPU.

### 1) Créer un Space

- Type: **Docker**
- Hardware: une instance **GPU** (selon disponibilité)

### 2) Fournir les fichiers Docker

Ce dossier contient un `Dockerfile` conçu pour:
- démarrer un serveur **vLLM OpenAI-compatible** (`/v1/chat/completions`)
- démarrer l’API FastAPI du POC (`/health`, `POST /triage`, `/audit/{id}`)
- configurer l’API pour utiliser le backend `vllm-openai`

Fichiers:
- `Dockerfile`
- `start.sh`

### 3) Variables d’environnement à définir dans le Space

Minimum:
- `HF_TOKEN` (si les modèles/adapters sont privés)

Optionnel:
- `VLLM_MODEL` (par défaut `Qwen/Qwen3-1.7B-Base`)

### 4) Endpoints

- Endpoint POC: `GET /health`, `POST /triage`, `GET /audit/{interaction_id}`
- Endpoint vLLM: `POST /v1/chat/completions`

### 5) Démonstration soutenance

1) Ouvrir `https://<space-url>/docs`
2) Tester `GET /health`
3) Tester `POST /triage` avec un message sans red-flags (ex: rhume, mal de gorge, gastro légère)

## Notes importantes

- Les adapters LoRA publiés sont sur Hugging Face: `perachon/p14-model`.
- En cloud, on peut soit:
  - utiliser directement l’adapter (si vLLM supporte LoRA selon votre version),
  - soit servir le modèle de base avec vLLM, et garder l’adapter côté API via Transformers+PEFT.

Dans ce repo, l’API supporte désormais un backend `vllm-openai` qui forward les générations vers vLLM.
