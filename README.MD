# NLP Project
## Sommaire
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f8/CY_Tech.svg" align="right"
     alt="Size Limit logo by Anton Lovchikov" width="170">
* [Présentation](#présentation) : Présentation du projet
* [Pré-requis](#pré-requis) : Installation de l'environnement et des dépendances
* [Outils utilisés](#outils-utilisés) : Librairies et outils de développement
* [Architecture du projet](#architecture-du-projet) : Structure des dossiers
* [Fonctionnalités principales](#fonctionnalités-principales) : Liste des fonctionnalités et scripts
* [Lancer l'application](#lancer-lapplication) : Instructions pour exécuter le projet en CLI ou UI


## Présentation

**NLP Project** est un projet axé sur le Traitement du Langage Naturel (NLP) et le Fine-Tuning de LLMs.

Contributeurs :
- Bari-Joris BOICOS
- Baptiste LAMBERT
- Jonathan NGO
- Tom CONTI


Ce projet permet de **configurer**, **entraîner** et **utiliser** un **adaptateur LoRA** pour spécialiser un modèle (notamment `Qwen2.5-1.5B-Instruct`) dans la **génération** et le **traitement** de **requêtes SQL**. Il dispose d'outils d'observation, de création de datasets, d'un agent conversationnel en ligne de commande et d'une interface graphique.

## Pré-requis

Le projet requiert **Python 3.11** et utilise **UV** comme gestionnaire de dépendances et d'environnements (en remplacement de `pip`). [UV](https://docs.astral.sh/uv/) est un équivalent de Pip écrit en Rust, offrant des performances accrues et une gestion automatique des environnements virtuels.

### Installation d'UV

> [!CAUTION]
> Ne confondez pas UV avec UVicorn !
> * **UV** : gestionnaire de dépendance
> * **UVicorn** : gestionnaire de serveur web utilisant UV
> Bien que les deux puissent être installés via pip, privilégiez les commandes officielles ci-dessous.

Sur **Windows** (PowerShell) :

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sur **Linux** :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Tester l'installation :

```bash
uv --version
```

### Initialisation du projet
Pour installer les dépendances et créer l'environnement virtuel (`.venv`), exécutez la commande suivante à la racine du projet :
```bash
uv sync
```

*(Commandes utiles supplémentaires : `uv add <lib>` pour ajouter, `uv remove <lib>` pour supprimer, `uv lock --upgrade` pour tout mettre à jour).*

## Outils utilisés

* **Python 3.11** - Langage de programmation
* **UV** - Gestionnaire de paquets et d'environnements
* **Torch** - Framework de Deep Learning
* **Pandas / Numpy** - Manipulation et analyse de données
* **Matplotlib** - Visualisation et tracés (ex: courbes de perte)
* **Trl / Scikit-learn** - Machine Learning et Fine-tuning (Reinforcement Learning)

## Architecture du projet

D'après l'arborescence, le code source est organisé ainsi :

```text
NLP_project/
├── qwen-sql-lora/          # Dossier contenant l'adaptateur pré-entraîné (best_adapter)
├── src/
│   ├── baseline_dataset/   # Datasets JSON (train/test/val) et scripts de base
│   ├── LoRA_fine_tuner/    # Scripts pour l'entraînement (LoRAFineTunner.py) et l'Agent SQL
│   ├── model_observation/  # Notebooks et démos (BPE, rapports d'observation)
│   └── ui/                 # Fichiers de l'interface graphique (UI.py)
├── agent.ipynb             # Notebook interactif pour l'Agent
├── pyproject.toml / uv.lock# Configuration et verrouillage des dépendances UV
└── README.md
```

## Fonctionnalités principales

### Génération de l'adaptateur LoRA
Le dépôt contient un adaptateur pré-entraîné situé dans `qwen-sql-lora/best_adapter`. La logique d'entraînement, le calcul des courbes de perte (loss curves) et la génération du modèle sont gérés de manière modulaire.

### Agent SQL intelligent
L'application propose un Agent interactif capable de comprendre et traiter des requêtes SQL en s'appuyant sur le modèle Fine-Tuné.

### Observation et Datasets
Plusieurs outils sont intégrés pour comprendre le modèle (comme l'encodage par paires d'octets - BPE) et gérer les jeux de données (création de *baselines*, séparation train/val/test).

## Lancer l'application
*Toutes les commandes d'exécution doivent utiliser `uv run` pour bénéficier de l'environnement virtuel.*

Vous pouvez directement lancer l'application depuis `agent.ipynb`

Le lancement peut être également être effectué directement en ligne de commandes via les méthodes ci-dessous : 

### Version graphique
Pour lancer l'**interface graphique** :
```bash
uv run python -m src.ui.UI
```

### Version terminale (CLI)

Pour lancer l'**Agent SQL en ligne de commande** :

```bash
uv run src/LoRA_fine_tuner/SQLAgent.py
```
