---
title: p14-space
emoji: "🩺"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# P14 — CHSA Triage API (CPU demo)

Ce Space expose une API FastAPI de démonstration.

- Health: `/health`
- Swagger: `/docs`
- Triage: `POST /triage`

Note: backend par défaut = `TRIAGE_BACKEND=stub` (pas de vLLM/GPU).
