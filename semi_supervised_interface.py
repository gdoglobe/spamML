import streamlit as st
from semi_supervised_algo import df_performance_metrics
from PIL import Image

# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>SSL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage semi-supervisé, un algorithme exploite de grandes quantités de données non étiquetées disponibles dans de nombreux cas d'utilisation en combinaison avec des ensembles de données étiquetées généralement plus petits.
""")
st.markdown(
"""
Pour notre étude, nous avons choisi trois algorithmes différents:
* Self Training Classifier : est basé sur l'algorithme de Yarowsky. En utilisant cet algorithme, un classificateur supervisé donné peut fonctionner comme un classificateur semi-supervisé et prédit des étiquettes pour les échantillons non étiquetés et ajoute un sous-ensemble de ces étiquettes à l'ensemble de données étiquetées.
* Label Propagation & Label Spreading : fonctionnent en construisant un graphique de similarité sur tous les éléments de l'ensemble de données d'entrée. Ils diffèrent par les modifications apportées à la matrice de similarité et par l'effet de serrage sur les distributions d'étiquettes.
""")
image = Image.open('picture/semisup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé de mails où pour une partie d'entre eux on connait leur nature et une partie pour laquelle on ne connait pas leur nature. Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.       
Pour chaque algorithme, nous allons l'entraîner et lui permettre de définir un modèle de prédiction grâce au jeu d'entraînement (training set).
Ensuite nous allons tester le modèle avec le jeu de test (test set) et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.dataframe(df_performance_metrics)