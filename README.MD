# 🤖 NLP Project

## Descriptif

- 📅 Date de rendu : 2026-09-30 (présentation)

## Environement

### Python et UV
- 🐍 Python 3.11
- 💻 UV : gestionnaire de dépendance (à la place de pip)

### Le gestionnaire de dépendances
On utilise [UV](https://docs.astral.sh/uv/). C'est un équivalent de Pip écrit en Rust, c'est donc beaucoup plus rapide.
De plus, UV propose une gestion automatique des dépendances et de leurs mises à jour mais aussi un outil de gestion environment python.

#### Installation d'UV

> [!caution]
> Ne confondez pas UV avec UVicorn ! 
> - UV : gestionnaire de dépendance
> - UVicorn : gestionnaire de serveur web utilisant UV  
> Les deux peuvent être installés via pip, mais passez plutôt par les commandes officielles

Sur Windows :
````shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
````

Sur linux:
````bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

Tester l'installation : 
````bash
uv --version
````

#### Les commandes utiles

Gestionnaire de dépendances :
````bash
# installer les dépendances et créer le .venv (à exécuté en premier après avoir installer UV !)
uv sync

# Ajouter une dépendance 
uv add <nom_de_la_librairie>

# Supprimer une dépendance 
uv remove <nom_de_la_librairie>

# Tout mettre à jour selon pyproject.toml
uv lock --upgrade

# Mettre à jour un package spécifique
uv lock --upgrade-package <nom_de_la_librairie>
````

Gestionnaire python :
````bash
# Installer une / des versions de python
uv python install <version>

# Changer de version de python pour le projet
uv pin python <version>
````

> [!warning]
> Les commandes UV ne changent pas la version de python du système, mais seulement celle du projet courant
> Ainsi, le compilateur et l'interpréteur python utilisés sont ceux du projet courant et non ceux du système.
>
> Ne changez pas la version de python du système sans prévenir !

### 📚 Librairies

- Torch
- Pandas
- Numpy

## ✅ Les bonnes pratiques de code

1. Respecter le plus que possible les conventions PEP 8 (Il y à des extensions VScode / JetBrains qui permettent de respecter automatiquement les conventions)
2. écrivez des commentaires clairs et précis ainsi que des Docstrings détaillés pour les classes et méthodes
3. Organisez le code par packages (en dossiers, comme sur Java. Attention : les imports ne sont qu'ascendants)
4. Faites des classes Python avec les Dunders methods importantes (\__str\__; \__repr\__)
4. Faites vos propres branches, ne jamais push sur "master" si ça ne marche pas chez vous !