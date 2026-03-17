# Étape 1 — Collecte, structuration, anonymisation (SFT/DPO)

Objectif : produire un dataset médical bilingue **prêt** pour :

- le **SFT** (≈ 5 000 paires instruction-réponse)
- l’alignement **DPO** (paires préférées/rejetées)

…tout en garantissant :

- **séparation** entraînement vs évaluation clinique
- **auditabilité** (trace de chaque transformation)
- conformité **RGPD** (anonymisation, justification)

## Format des données

### SFT (instruction → réponse)

Fichier JSONL : chaque ligne suit le schéma (voir aussi `SFTRecord`) :

- `instruction` (str)
- `input` (str, optionnel)
- `output` (str)
- `lang` (`fr`/`en`)
- métadonnées : `source`, `symptoms`, `history`, `vitals`, `confidence`, `pii_redacted`

Exemples : [data/sample/sft_sample.jsonl](data/sample/sft_sample.jsonl)

### DPO (préférences)

Fichier JSONL : chaque ligne suit le schéma (voir aussi `DPORecord`) :

- `prompt` (str)
- `chosen` (str) : réponse préférée
- `rejected` (str) : réponse non préférée
- `lang` (`fr`/`en`)
- métadonnées : `source`, `confidence`, `pii_redacted`

Exemples : [data/sample/dpo_sample.jsonl](data/sample/dpo_sample.jsonl)

## Pipeline fourni dans le repo

Le builder produit :

- `sft.jsonl`, `dpo.jsonl`
- `metadata_schema.json`
- `splits/` : `*_train.jsonl`, `*_val.jsonl`, `*_test.jsonl`
- `audit_log.jsonl` : journal append-only des étapes
- `hf/` : export Hugging Face Datasets via `save_to_disk` (optionnel)

Le dossier `data/processed/` est ignoré par Git : seules de petites données “sample” sont commit.

### Commande (sur les samples)

Sur Windows, avec ton `.venv` :

```bash
 d:/Pro/Formations/OpenClassrooms/AIEngineer/projets/14/mission/.venv/Scripts/python.exe -m scripts.build_datasets \
   --input_dir data/sample \
   --out_dir data/processed \
   --seed 42 \
   --clinical_eval_dir data/sample
```

### Options utiles

- `--anonymize` : active Presidio
- `--anonymize_operator` : `replace|mask|redact`
- `--no_export_hf` : désactive l’export `save_to_disk`
- `--clinical_eval_dir` : copie un jeu d’éval clinique **séparé** (pas de mélange)

## RGPD & anonymisation

Gabarit de justification : [data/RGPD_PROCESS.md](data/RGPD_PROCESS.md)

Sources/licences à documenter : [data/SOURCES.md](data/SOURCES.md)

## Point important (consigne)

- Ne pas mélanger entraînement et évaluation : utiliser un dossier séparé (ex: `data/eval/`).
- Garder une trace de chaque transformation : `audit_log.jsonl` + hashes.
