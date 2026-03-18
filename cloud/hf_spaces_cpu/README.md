---
title: p14-space
emoji: "🩺"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# HF Spaces (gratuit / CPU Basic) — endpoint FastAPI (sans vLLM)

Objectif: obtenir un **endpoint cloud de démonstration** gratuit (CPU) pour l’API du POC.

Important: ce déploiement **n’utilise pas vLLM** (vLLM requiert un GPU pour être pertinent/rapide). Le rapport explique l’option vLLM (GPU) comme voie recommandée si budget.

## Ce que ce Space fournit

- `GET /health`
- `POST /triage`
- `GET /audit/{interaction_id}`
- Swagger: `/docs`

Backend par défaut: `TRIAGE_BACKEND=stub` (ne charge pas de modèle).

## Déploiement

Dans ton Space `perachon/p14-space` (SDK Docker, template Blank, CPU Basic):

0) Crée un `README.md` **à la racine du Space** avec le front-matter YAML Spaces (obligatoire).
	- Template prêt à copier: `cloud/hf_spaces_cpu/README.space_root.md` (à coller dans `README.md` du Space)

1) Mets un `Dockerfile` à la racine du Space.
2) Mets `start.sh` à la racine.
3) Copie `src/` (le package `triage_llm`) dans le Space.
4) Copie `requirements-api.txt` dans le Space.

Astuce: le plus simple est de copier/coller le contenu des fichiers de ce dossier (`cloud/hf_spaces_cpu/`) vers la racine du repo du Space.

## URL runtime

Quand le Space est "Running", l’URL publique est généralement:

- `https://perachon-p14-space.hf.space`

Puis:

- `https://perachon-p14-space.hf.space/health`
- `https://perachon-p14-space.hf.space/docs`

## Test rapide (PowerShell)

```powershell
$base = "https://perachon-p14-space.hf.space"
Invoke-RestMethod "$base/health" | ConvertTo-Json -Depth 6

$payload = @{ patient_message = "J'ai mal à la gorge depuis 2 jours, nez qui coule, pas de fièvre."; lang = "fr"; context = @{} }
$json = ($payload | ConvertTo-Json -Depth 6)
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod "$base/triage" -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes | ConvertTo-Json -Depth 10
```
