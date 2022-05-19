import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import pickle

# recover an exemple
file = open('pkl/home_page_exemple.pkl', 'rb')
text = pickle.load(file)
text1 = pickle.load(file)
text2 = pickle.load(file)
file.close()


# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>Home Page<h1>", unsafe_allow_html=True)
st.markdown(
"""
Au travers d'un exemple illustré, Ce site a pour objectif de déterminer si l'utilisation de l'apprentissage semi-supervisé se révèle être plus performant que celui de l'apprentissage supervisé et non-supervisé.
""")

dataset = pd.read_csv("spam_clean.csv")
shape=dataset.shape
count_Class=pd.value_counts(dataset["label_num"], sort= True)
labels = 'Not-Spam', 'Spam'
sizes = [count_Class[0],count_Class[1]]
explode = (0.1, 0) 
st.markdown("<h1 style='text-align: center; color: red;'>Le Dataset<h1>", unsafe_allow_html=True)
st.write("Le dataset choisi pour réaliser cette étude est un ensemble de mail composé de spam et de mail traditionnel. Le dataset est composé de",shape[0],"mails dont ",sizes[0]," non spam et ",sizes[1]," spam.")


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90,colors= ["grey", "red"])
ax1.axis('equal') 
ax1.set(facecolor = "orange")

data = {labels[0]:sizes[0], labels[1]:sizes[1]}
Courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(Courses, values,color= ["grey", "red"])


col1, col2, col3 = st.columns([1, 0.7, 0.1])
col1.pyplot(fig)
col2.pyplot(fig1)

st.markdown(
"""
Les mails en machine learning représentent des données textuels. Ils doivent donc être soumis à une période de prétraitement.
Cette étape permet de réduire de manière significative la taille des documents textuels d'entrée et d’améliorer l’analyse des données.  
Le prétraitement est généralement divisé en différentes parties : le stop word removal, le stemming, la lementisation et la tokenisation.  
Pour ce faire, nous allons utiliser la fonction suivante et montrer son fonctionnement en utilisant un mail du dataset.
""")
image = Image.open('picture/clean.png')
st.image(image)
st.markdown("<u>Voici un exemple de mail présent dans le dataset :</u>", unsafe_allow_html=True)
st.markdown("<i>"+text+"</i>",unsafe_allow_html=True)



st.title("Stopwords")


st.markdown(
"""
La première manipulation souvent effectuée dans le traitement du texte est la suppression de ce qu'on appelle en anglais les stopwords. 
Ce sont les mots très courants dans la langue étudiée ("and", "at", "so") qui n'apportent pas de valeur informative pour la compréhension du "sens" d'un document et corpus. 
Il sont très fréquents et ralentissent notre travail.  
Afin de réduire d'avantage la taille des documents et de les uniformiser, nous enlevons également les majuscules, chiffres, carcatères spéciaux et les mots de petites tailles.
""")
st.markdown("<u>Après cette étape, le mail devient :</u>", unsafe_allow_html=True)
st.markdown("<i>"+text1+"</i>", unsafe_allow_html=True)

st.title("Lemmatisation & Stemming")

st.markdown(
"""
- Le processus de « lemmatisation » consiste à représenter les mots  sous leur forme canonique. Par exemple, pour un verbe, ce sera son infinitif. L'idée étant encore une fois de ne conserver que le sens des mots utilisés dans le corpus.  
- Le processus de racinisation(ou stemming en anglais) consiste à ne conserver que la racine des mots étudiés. L'idée étant de supprimer les suffixes, préfixes et autres des mots afin de ne conserver que leur origine
"""
)
st.markdown("<u>Une fois ces processus réalisés, le mail se présente ainsi :</u>", unsafe_allow_html=True)


st.markdown("<i>"+text2+"</i>", unsafe_allow_html=True)


st.title("Vecteurs numériques")

st.markdown(
"""
La dernière phase consiste à transformer des données textuelles en vecteurs numériques avec TF-IDF.
Le TF-IDF calcule des «poids» qui représentent l'importance d'un mot pour un document dans une collection de documents.  
La valeur TF-IDF augmente proportionnellement au nombre de fois qu'un mot apparaît dans le document et est compensée par le nombre de documents du corpus qui contiennent le mot.
"""
)