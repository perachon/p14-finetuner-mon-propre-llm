---
title: "Rapport technique — POC Agent IA de triage médical (CHSA)"
subtitle: "Version condensée de rendu"
author: ""
date: ""
---

# Rapport technique — POC Agent IA de triage médical (CHSA)

**Projet 14 — Fine-tunez votre propre LLM (OpenClassrooms)**

- **Livrable attendu** : PDF, 20 pages maximum
- **Nature du projet** : Proof of Concept technique
- **Important** : ce prototype ne constitue pas un dispositif médical et ne remplace pas un professionnel de santé

---

## Sommaire

1. Résumé exécutif  
2. Contexte, besoins et contraintes  
3. Données  
4. Modèle et entraînement  
5. Évaluation et sécurité  
6. Déploiement et validation  
7. CI/CD  
8. Recommandations et passage à l’échelle  
9. Annexes utiles  

---

# 1) Résumé exécutif

Le Centre Hospitalier Saint-Aurélien souhaite disposer d’un agent IA capable d’assister le triage initial des patients afin de mieux prioriser les cas aux urgences. L’objectif de ce projet est de démontrer la **faisabilité technique** d’un tel système à travers un POC complet couvrant la préparation des données, la spécialisation d’un LLM, l’exposition d’une API, la traçabilité des interactions, la CI/CD et une démonstration cloud.

Le prototype s’appuie sur un dataset médical bilingue, un modèle `Qwen/Qwen3-1.7B-Base` spécialisé en deux temps (**SFT avec LoRA**, puis **DPO**), une API FastAPI de triage, un audit SQLite et un endpoint public sur Hugging Face Spaces. Un garde-fou de sécurité basé sur des **red flags** permet de court-circuiter le modèle sur les cas potentiellement graves afin de privilégier la prudence.

Les principaux résultats obtenus sont les suivants :

- dataset SFT : **5 500** exemples ; dataset DPO : **3 000** paires ;
- splits reproductibles train / validation / test ;
- adapters LoRA publiés sur Hugging Face ;
- API de triage fonctionnelle avec audit ;
- endpoint cloud public de démonstration ;
- pipeline GitHub Actions opérationnel ;
- suivi des expérimentations via **MLflow local** et courbes de loss / eval_loss exportées.

Les mesures de latence montrent un écart net entre la démo API légère et l’inférence locale du vrai modèle, ce qui confirme l’intérêt d’une trajectoire de déploiement plus performante via **vLLM** sur Linux + GPU. Le POC est donc concluant comme démonstrateur technique supervisé, mais il ne peut pas être considéré comme un outil clinique autonome sans validation métier et gouvernance renforcée.

### Mesures de latence

| Environnement | Backend | Matériel | P50 (s) | P95 (s) |
|---|---|---|---:|---:|
| Local | transformers+peft | RTX 4050 6GB | 13.491 | 13.693 |
| Local | stub | CPU | 0.020 | 0.042 |
| Cloud | stub | HF Spaces CPU | 0.227 | 0.306 |
| Cloud | vLLM | Linux + GPU | — | — |

---

# 2) Contexte, besoins et contraintes

Le CHSA fait face à une surcharge régulière de son service d’urgences. Le besoin métier est donc de disposer d’un assistant capable de **recueillir les symptômes**, **proposer un niveau de priorité**, **expliquer la recommandation**, **préparer une intégration au SIH** et **garantir la traçabilité** des interactions.

Dans le cadre du POC, ces besoins sont couverts de la manière suivante :

- entrée patient en **français ou anglais** ;
- questions de suivi générées par l’API selon le message et le champ `context` ;
- décision de triage sur trois niveaux : `urgence_maximale`, `urgence_moderee`, `urgence_differee` ;
- explications et recommandations prudentes ;
- API REST compatible avec une intégration à un système tiers ;
- audit complet des échanges.

Le projet reste soumis à plusieurs contraintes fortes :

- pas de diagnostic certain ni de posologie ;
- pas de données issues d’un SIH réel ;
- environnement local Windows avec GPU limité ;
- démonstration cloud gratuite, donc sans déploiement GPU public effectif ;
- nécessité de garder un prototype reproductible, testable et traçable.

Le périmètre retenu est donc celui d’un **prototype technique crédible**, et non d’un outil clinique prêt à l’emploi.

---

# 3) Données

La stratégie retenue a été de s’appuyer sur une source principale unique, clairement documentée et directement exploitable : `cyrille-elie/CHSA-Triage-Medic-Full-Dataset` sur Hugging Face, sous licence **MIT**. Ce choix permettait de disposer rapidement d’un corpus bilingue cohérent, de limiter les ambiguïtés liées à la fusion de sources multiples, et de concentrer l’effort sur la chaîne de traitement complète du POC.

Deux sous-configurations ont été utilisées :

- `sft_medical_dataset` pour le fine-tuning supervisé ;
- `dpo_dataset` pour l’alignement par préférences.

Les données ont été converties en JSONL, enrichies avec des métadonnées utiles au projet, puis redivisées en ensembles train / validation / test avec une seed fixe. Le projet produit ainsi des artefacts clairement séparés pour l’entraînement supervisé et pour le DPO, ainsi qu’un journal des transformations.

Artefacts principaux :

- `data/processed/sft.jsonl` : **5 500** exemples ;
- `data/processed/dpo.jsonl` : **3 000** paires ;
- `data/processed/splits/` ;
- `data/processed/metadata_schema.json` ;
- `data/processed/audit_log.jsonl`.

La démarche RGPD est documentée, mais aucune donnée patient issue d’un système hospitalier réel n’a été utilisée. Une stratégie d’anonymisation via Presidio est prévue si nécessaire, mais le POC repose ici sur une source publique déjà documentée.

---

# 4) Modèle et entraînement (Étape 2)

Le modèle de base choisi est `Qwen/Qwen3-1.7B-Base`. Ce choix repose sur un compromis entre qualité, compacité et faisabilité matérielle. L’objectif n’était pas de maximiser la taille du modèle, mais de démontrer qu’un LLM pouvait être spécialisé pour le triage médical dans un environnement réaliste de projet.

L’entraînement s’est déroulé en deux étapes :

1. **SFT avec LoRA** pour apprendre au modèle à produire des réponses plus structurées, prudentes et orientées triage.
2. **DPO** pour améliorer le comportement du modèle en lui apprenant à préférer certaines réponses à d’autres.

Le recours à **LoRA** permet de ne pas réentraîner l’intégralité du modèle, mais uniquement des couches d’adaptation légères. Cette approche est particulièrement adaptée à un POC sous contrainte GPU, car elle réduit fortement le coût mémoire et facilite la publication des artefacts finaux.

Configurations principales :

- **SFT par défaut** : batch size 1, gradient accumulation 8, learning rate `2e-4`, `max_seq_length=1024`, LoRA `r=16`, `alpha=32`, `dropout=0.05` ;
- **runs longs low-VRAM** : `max_seq_length=128`, `max_steps=800`, LoRA `r=4`, `alpha=8`, modules ciblés `q_proj` / `v_proj`, `fp16` ;
- **DPO par défaut** : `beta=0.1`, `max_length=256`, learning rate `1e-5`, batch size 1 ;
- **runs longs DPO** : `max_length=128`, `max_steps=400`.

Les artefacts produits sont conservés localement dans `checkpoints/` et publiés sur Hugging Face sous forme d’**adapters LoRA** dans `perachon/p14-model`. Le “modèle final” du POC correspond donc au modèle de base Qwen3-1.7B combiné avec les adapters appris pendant le SFT et le DPO.

Le suivi des expérimentations est désormais assuré via **MLflow local**, avec export de courbes de `loss` et `eval_loss`. Cette brique complète le projet sur le volet demandé par le mentor : les runs sont maintenant suivis, visualisables et documentés.

---

# 5) Évaluation et sécurité

L’évaluation réalisée dans ce projet est une **évaluation de POC technique**, et non une validation clinique complète. L’objectif est de vérifier que le système fonctionne de bout en bout, que ses sorties restent cohérentes avec le cas d’usage de triage, et que des mécanismes de sécurité simples limitent les comportements les plus risqués.

La démarche d’évaluation repose sur trois niveaux :

- validation technique : schémas, API, tests, CI ;
- validation fonctionnelle : cas de triage en français et en anglais ;
- validation opérationnelle : mesures de latence.

La sécurité du prototype repose principalement sur une logique de **red flags**. Lorsqu’un message contient un indice de gravité, le système :

- force la priorité à `urgence_maximale` ;
- renvoie une recommandation prudente ;
- évite l’appel au modèle génératif ;
- conserve la trace dans l’audit.

Cette logique constitue une première barrière robuste sur des cas critiques comme la douleur thoracique, l’évanouissement ou le saignement important. Elle n’est pas exhaustive, mais elle démontre qu’un assistant génératif peut être encadré par des règles métier explicites dans un contexte sensible.

Des tests automatisés vérifient notamment :

- la disponibilité de l’API ;
- la validité des échantillons SFT / DPO ;
- la détection de red flags en français et en anglais.

Enfin, chaque interaction est stockée et relisible, ce qui fait de l’audit un élément de l’évaluation autant qu’un élément technique du produit.

---

# 6) Déploiement et validation (Étape 3)

Le prototype est exposé sous forme d’une **API FastAPI**. Le but n’est pas seulement de faire tourner un modèle, mais de fournir un service démontrable, testable et intégrable, avec un contrat d’API clair, une route de triage et une capacité d’audit.

Endpoints principaux :

- `GET /health` ;
- `POST /triage` ;
- `GET /audit/{interaction_id}`.

L’architecture prévoit plusieurs backends d’inférence :

- `stub` pour la démonstration rapide et la CI ;
- `transformers+peft` pour l’inférence locale avec le vrai modèle ;
- `vllm-openai` pour un forwarding vers un serveur vLLM OpenAI-compatible.

Ce choix permet de conserver le même contrat d’API tout en changeant le moteur selon le contexte d’exécution. Il rend le prototype à la fois démontrable localement, testable automatiquement et préparé pour une future infrastructure plus performante.

Le projet inclut également un endpoint public de démonstration sur Hugging Face Spaces :

- Space : https://huggingface.co/spaces/perachon/p14-space
- URL API : https://perachon-p14-space.hf.space
- Swagger : https://perachon-p14-space.hf.space/docs

Ce Space public démontre le contrat d’API, les garde-fous et la traçabilité. En revanche, il ne s’agit pas d’un déploiement vLLM GPU public. vLLM reste ici la **voie d’industrialisation préparée**, avec packaging Docker et backend compatible, mais non activée en ligne dans le cadre gratuit retenu pour le POC.

La traçabilité repose sur un stockage SQLite dans `runs/audit/audit.db`, ce qui permet de retrouver les interactions et de prouver le comportement du système lors de la soutenance.

---

# 7) CI/CD

Le projet intègre un pipeline **GitHub Actions** afin d’automatiser les contrôles essentiels du dépôt. Même dans le cadre d’un POC, cette chaîne est importante : elle réduit le risque de régression, garantit que le projet reste exécutable dans un environnement propre, et renforce sa crédibilité technique.

Le workflow met en place deux grands contrôles :

- vérification du code avec `ruff check .` ;
- exécution des tests avec `pytest -q` ;
- build Docker pour vérifier la conteneurisation.

Cette chaîne CI/CD constitue une base solide d’intégration continue. Elle n’automatise pas encore un redéploiement complet de l’endpoint cloud ni une republication des modèles, mais elle répond bien à l’attendu du projet sur l’automatisation des contrôles et la maintenabilité du prototype.

---

# 8) Recommandations et passage à l’échelle

Le POC démontre la faisabilité technique du système, mais une montée en maturité exigerait plusieurs étapes complémentaires.

### Court terme

- constituer un jeu d’évaluation clinique relu par un professionnel de santé ;
- enrichir la couverture des red flags et mieux gérer les formulations ambiguës ;
- renforcer la validation stricte des sorties côté API ;
- poursuivre les mesures de latence et de stabilité.

### Moyen terme

- activer une infrastructure Linux + GPU avec **vLLM** pour réduire la latence ;
- renforcer le monitoring du service et l’analyse des cas à risque ;
- structurer davantage le suivi des expérimentations et le versionnement des artefacts ;
- préparer une intégration supervisée avec un portail ou un SIH.

La recommandation stratégique n’est donc pas un passage direct en production, mais une **phase pilote supervisée**. Le prototype constitue une base crédible pour cette étape, à condition de conserver une validation humaine, une gouvernance renforcée et un cadre de sécurité maîtrisé.

---

# 9) Annexes utiles

## 9.1 Livrables publics

- Code source : https://github.com/perachon/p14-finetuner-mon-propre-llm
- Space Hugging Face : https://huggingface.co/spaces/perachon/p14-space
- API publique : https://perachon-p14-space.hf.space
- Swagger : https://perachon-p14-space.hf.space/docs
- Modèle (adapters LoRA) : https://huggingface.co/perachon/p14-model

## 9.2 Artefacts internes importants

- Données traitées : `data/processed/`
- Splits : `data/processed/splits/`
- Audit log données : `data/processed/audit_log.jsonl`
- Checkpoints : `checkpoints/`
- Audit API : `runs/audit/audit.db`
- Tracking MLflow : `runs/experiment_tracking/mlruns/`
- Courbes d’entraînement : `runs/experiment_tracking/curves/`

## 9.3 Commandes clés

Construction des datasets :

```bash
python -m scripts.build_datasets --input_dir data/sample --out_dir data/processed --seed 42
```

SFT :

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

Runs longs :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_long_experiments.ps1 -SftMaxSteps 800 -DpoMaxSteps 400 -SeqLen 128
```

## 9.4 Démonstration soutenance

Ordre conseillé :

1. `GET /health`
2. `POST /triage` sur un cas standard
3. `POST /triage` sur un cas avec red flag
4. `GET /audit/{interaction_id}` pour montrer la traçabilité

Cette démonstration permet de montrer en quelques minutes la disponibilité du service, la structure de la réponse, la logique de sécurité et l’audit.