# Rapport technique — POC Agent IA de triage médical (CHSA)

> Contraintes : PDF, 20 pages max. Mission uniquement.

## 1. Résumé exécutif (1 page)

- Problème (surcharge urgences), objectifs du POC.
- Résultats clés (métriques, latence, limites).
- Recommandations go/no-go + roadmap.

## 2. Contexte & exigences

- Questionnaire adaptatif, priorisation (max/modérée/différée), explications.
- Intégration SI, traçabilité/audit.
- Contraintes : RGPD, données, sécurité, supervision clinique.

## 3. Données (Étape 1)

### 3.1 Inventaire des sources

- MediQA / FrenchMedMCQA / MedQuAD / UltraMedical-Preference.
- Licences et conditions d'usage.

### 3.2 Schéma des données

- Champs, formats (JSONL / HF Datasets), splits (train/val/test), jeux d'évaluation cliniques séparés.

### 3.3 Anonymisation & RGPD

- Méthode (Presidio), stratégies (replace/mask/redact).
- Contrôles qualité (échantillonnage, tests automatiques, audit).

## 4. Modèle & entraînement (Étape 2)

### 4.1 Modèle de base

- Qwen3-1.7B-Base : justification.

### 4.2 SFT + LoRA

- Hyperparamètres, ressources, temps d'entraînement.
- Checkpoints, seeds, reproductibilité.

### 4.2bis Runs “courts” vs “longs” (méthodologie)

- Run court : valider le pipeline (dataset, tokenization/troncature, loss, pas d’OOM) et obtenir un premier adapter.
- Run long : augmenter le budget de steps (même config) pour améliorer la qualité.
- Comparaison : conserver des dossiers de sortie distincts (ex: `*_LONG_*`) et reporter les métriques (eval_loss, stabilité, NaN).
- Traçabilité : lister les artefacts (dossiers `checkpoints/`) et la publication éventuelle sur Hugging Face (repo modèle, adapters).

### 4.3 Alignement DPO

- Construction des paires (chosen/rejected).
- Paramètres DPO (beta, longueurs).

## 5. Évaluation & sécurité

- Métriques automatiques et cliniques.
- Tests d'hallucinations / recommandations dangereuses.
- Analyse d'erreurs et limites.

## 6. Déploiement (Étape 3)

- Architecture (Docker, FastAPI, vLLM).
- Traçabilité (audit), gestion des secrets.
- Latence/robustesse, plan de surveillance.

## 7. Recommandations & passage à l'échelle

- Données supplémentaires, modèles 32B+, gouvernance.
- Plan d'industrialisation (MLOps, CI/CD, monitoring, sécurité).

## Annexes

- Détails des prompts, schémas, configurations.
