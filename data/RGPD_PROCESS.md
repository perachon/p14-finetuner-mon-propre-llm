# Processus RGPD & anonymisation (gabarit)

Ce document sert de **justification** et de **traçabilité** sur la conformité RGPD du POC.

## 1) Périmètre

- Nature des données : textes médicaux (questions/réponses, consignes), paires préférentielles.
- Exclusion : aucune donnée patient issue du SIH réel n'est intégrée au POC.

## 2) Inventaire des sources

Référez-vous à `data/SOURCES.md`.

Pour chaque source : préciser la licence, les conditions d'usage, et le mode d'acquisition.

## 3) Stratégie d'anonymisation

- Outil : Presidio (AnalyzerEngine + AnonymizerEngine)
- Entités ciblées (à adapter) : PERSON, PHONE_NUMBER, EMAIL_ADDRESS, LOCATION
- Stratégies : `replace` / `mask` / `redact`

Commande recommandée (exemple) :

```bash
python -m scripts.build_datasets --input_dir data/raw --out_dir data/processed --anonymize
```

## 4) Contrôles qualité

- Tests automatiques : échantillonnage + vérification absence de patterns (emails, téléphones).
- Revue manuelle : contrôle sur un échantillon (ex: 200 lignes).
- Archivage : journal des transformations `data/processed/audit_log.jsonl`.

## 5) Séparation entraînement / évaluation

- Les jeux d'évaluation clinique sont stockés séparément (ex : `data/eval/`) et ne sont pas mélangés aux splits train/val/test.

## 6) Conservation & accès

- Données brutes : accès restreint, non committées dans Git.
- Données traitées : versionnées (hashes + audit log), partageables si licence le permet.

## 7) Risques & limites

- Anonymisation imparfaite possible : documenter les taux de détection et les contrôles.
- Le POC ne doit pas être utilisé sans supervision clinique.
