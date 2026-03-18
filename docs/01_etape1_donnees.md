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

## Si tu n'as pas de données locales (recommandé : sources HF)

Si tu n'as pas (encore) de fichiers dans `data/raw/`, le plus simple est de partir de **datasets publics** sur Hugging Face, puis de les **convertir** vers le format JSONL attendu par ce repo (SFT/DPO).

Point clé : avant d'ingérer une source, vérifie **la licence** et la présence de **PII** (données personnelles). Même si ton dépôt HF est privé, les conditions de licence restent applicables.

### Lister rapidement des sources candidates (licence + langues)

Un helper est fourni : `scripts/hf_list_datasets.py`.

Exemples :

```bash
 .\.venv312\Scripts\python.exe scripts\hf_list_datasets.py --query "french medical" --limit 50 --languages fr,en
 .\.venv312\Scripts\python.exe scripts\hf_list_datasets.py --query "triage" --limit 50 --languages fr,en
 .\.venv312\Scripts\python.exe scripts\hf_list_datasets.py --query "med mcqa" --limit 50 --languages fr,en --out runs/hf_sources.md
```

Ensuite, reporte dans `data/SOURCES.md` les sources retenues (ID HF, URL, licence, transformations).
