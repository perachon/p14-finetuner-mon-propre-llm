# Étape 0 — Bases théoriques (SFT / DPO)

Ce document résume les notions utilisées dans le projet.

## SFT — Supervised Fine-Tuning

Le **Supervised Fine-Tuning (SFT)** consiste à repartir d’un modèle pré-entraîné (objectif « prédire le prochain token ») et à l’entraîner sur un jeu d’exemples **supervisés** (paires instruction → réponse) afin de :

- améliorer le **suivi d’instructions**,
- ancrer des **bonnes pratiques de réponse** sur un domaine (ici médical/triage),
- réduire les sorties hors-sujet.

Dans ce projet, le SFT est réalisé avec **LoRA** (adaptation de rang faible) afin de limiter le coût GPU et de ne pas fine-tuner tous les poids.

## DPO — Direct Preference Optimization

Le **Direct Preference Optimization (DPO)** est une méthode d’**alignement par préférences**.

Principe : on dispose de paires (prompt, réponse préférée, réponse rejetée) et on entraîne le modèle à donner plus de probabilité à la réponse préférée **sans utiliser de modèle de récompense intermédiaire**.

Objectif dans le POC : améliorer la conformité aux protocoles et **réduire les réponses dangereuses** (recommandations à risque, minimisation de red flags, etc.).

## Pourquoi faire un “sanity check” d’entraînement ?

Avant un run long/coûteux, on fait souvent un run très court (quelques minutes) pour valider :

- le chargement du modèle / tokenizer,
- la lecture des datasets,
- la boucle d’entraînement (loss qui baisse, pas d’OOM),
- la traçabilité (logs, checkpoints).

Ce n’est généralement pas « exigé » explicitement, mais c’est **très utile** pour gagner du temps et c’est un bon point à expliquer en soutenance (méthodologie rigoureuse).
