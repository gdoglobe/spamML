# Machine Learning Project

> Ce projet est une analyse technique cherchant à comparer l'apprentissage supervisé, non supervisé et semi-supervisé afin de determiner la performance de l'apprentissage semi-supervisé.

## Project Description

> Pour notre analyse nous nous sommes appuyés sur différents algorithmes supervisés, non supervisés et semi-supervisés :

- Supervisés:
  - LogistiqueRegression
  - MultinomialNB
  - KNeighborsClassifier
- Non Supervisés:
  - KMeans
  - MiniBatchKMeans
  - AgglomerativeClustering
- Semi-Supervisés:
  - Self Training Classifier avec LogistiqueRegression
  - Self Training Classifier avec MultinomialNB
  - Self Training Classifier avec KNeighborsClassifier
  - LabelSpreading
  - LabelPropagation

## Project Description

> Notre étude de cas s'appuye sur un dataset de mails composé de hams et de spams. Nous nous sommes appuyés sur différents indicateurs pour pouvoir déterminer si l'apprentissage semi-supervisé et aussi performant que l'apprentissage supervisé.
> Nous avons également crée une application capable de déterminer si un mail est un spam ou non. Il suffit simplement de choisir un mail et un algorithme.

## Prérequis

La présence de [Python](https://www.python.org/) sur la mchine est nécessaire.

## Build Setup

Après avoir cloner ce projet, vous devez executer les commandes suivantes :

```bash
# Installation de la bibliothèque streamlit
$ python -m pip install streamlit
# Installation de la bibliothèque streamlit_option_menu
$ python -m pip install streamlit_option_menu
# Installation de la bibliothèque pandas
$ python -m pip install pandas
# Installation de la bibliothèque nltk
$ python -m pip install nltk
# Installation de la bibliothèque matplotlib
$ python -m pip install matplotlib
# Installation de la bibliothèque sklearn
$ python -m pip install sklearn
```

## Démonstration

> Nous avons également mis en place un [Dashboard](https://developer.mozilla.org/fr/docs/Web/JavaScript) en ligne permettant de mieux comprendre les concepts d'apprentissage supervisé, non supervisé et semi-supervisé et de pouvoir les tester dans le cas d'une classification entre SPAMS & HAMS.

## GIT Commands

```bash
# Switch branches or restore working tree files
$ git checkout <name>

# Fetch from and integrate with another repository or a local branch
$ git pull

# Add all files contents to the index
$ git add *

# Record changes to the repository with clean message
$ git commit -m "la description du commit" 

# Update remote refs along with associated objects
$ git push

# Stash the changes in a dirty working directory away
$ git stash
```

Pour plus d'explication concernant le fonctionnement de cet API vous pouvez vous référez
aux documentations officielles ci-dessous :

- [Python docs](https://docs.python.org/3/).
- [SKLearn docs](https://scikit-learn.org/stable/).
- [Pandas docs](https://pandas.pydata.org/).
- [streamlit docs](https://streamlit.io/)

## Contributors

David Serruya, Steeven Alliel, Hanna Naccache, Gerard Dogolbe.
