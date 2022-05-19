from unsupervised_algo import df_performance_metrics
from PIL import Image
import streamlit as st


# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>UL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage non-supervisé, un algorithme reçoit un ensemble de données non-étiquetées sur lequel il va pouvoir s’entraîner et identifier des similarités afin de réaliser des regroupements.
""")
st.markdown(
"""
Pour notre étude, nous avons choisi trois algorithmes différents:
* K-moyennes (K-means) : est une méthode de quantification vectorielle , issue du traitement du signal , qui vise à partitionner n observations en k clusters dans lesquels chaque observation appartient au cluster dont la moyenne est la plus proche.
* Mini Batch K-means : est une alternative à l'algorithme de K-means. Son avantage est de réduire le coût de calcul en n'utilisant pas l'ensemble des données à chaque itération mais un sous-échantillon de taille fixe.
* Agglomerative Clustering (AC) : effectue un regroupement hiérarchique en utilisant une approche ascendante. Chaque observation commence dans son propre cluster, et les clusters sont successivement fusionnés.
""")
image = Image.open('picture/unsup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé de mails où on ne connait pas leur nature. Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.    
Pour chaque algorithme, nous allons l'entraîner avec le jeu d'entrainement (training set) et lui permettre de trouver des similarités afin de définir deux groupes.
Ensuite nous allons tester le modèle avec le jeu de test (test set) et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.write("")
st.dataframe(df_performance_metrics)