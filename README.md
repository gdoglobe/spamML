# Machine Learning Project

> Ce projet consiste à effectuer une analyse technique de la comparaison entre l'apprentissage supervisé, non supervisé et semi-supervisé afin de determiné si l'apprentissage semi-supervisé est aussi performant que l'apprentissage supervisé.
> Il a été réaliser par Steeven Alliel Gerard Dogolbe Hanna Naccache David Serruya © 2022.

## Project Description

> Pour effectuer notre analyse nous avons du tester et expliquer différents algorithmes de l'apprentissage supervisé, non supervisé et semi-supervisé :

- Supervisé
  - LogistiqueRegression
  - MultinomialNB
  - KNeighborsClassifier
- Non Supervisé
  - KMeans
  - MiniBatchKMeans
  - AgglomerativeClustering
- Semi-Supervisé
  - LogistiqueRegression
  - MultinomialNB
  - KNeighborsClassifier
  - LabelSpreading
  - LabelPropagation

## Project Description

> Pour parfaire notre travail, nous avons réalisé une etude de cas sur un dataset d'email dans lequel nous avons appliqué l'apprentissage semi-supervisé afin de classifier les données en deux groupes : SPAMS & HAMS.

## Prérequis

c'est obligatoire d'avoir [Python](https://www.python.org/) d'installer sur sa machine:

## Build Setup

Après avoir cloner ce projet, vous devez executer les commandes suivantes :

```bash
# Installation de la dépendance streamlit
$ python -m pip install streamlit
# Installation de la dépendance pandas
$ python -m pip install pandas
# Installation de la dépendance sklearn
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
