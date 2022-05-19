from PIL import Image
import streamlit as st
from supervised_algo import df_performance_metrics

# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>SL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage supervisé, un algorithme reçoit un ensemble de données étiquetées sur lequel il va pouvoir s’entraîner et définir un modèle de prédiction. Cet algorithme pourra par la suite être utilisé sur de nouvelles données afin de prédire leurs valeurs de sorties correspondantes.
""")
st.markdown(
"""
Pour notre étude, nous avons choisi trois algorithmes différents:
* La régression logistique : est un modèle de classification linéaire qui est le pendant de la régression linéaire, quand Y ne doit prendre que deux valeurs possibles (0 ou 1).
* Classification naïve bayésienne (Naive Bayes) : est une méthode de classification statistique qui peut être utilisée pour prédire la probabilité d'appartenance à une classe, dans notre cas il existe deux classes : Spam ou Non-Spam.
* K plus proches voisins (KNN) : suppose que des objets similaires existent à proximité dans cet espace (plus proches voisins).  En d'autres termes, des choses similaires sont proches les unes des autres. 
""")
image = Image.open('picture/sup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé mails pour lesquels on connait leur nature (spam ou non spam). Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.    
Pour chaque algorithme, nous allons l'entraîner et lui permettre de définir un modèle de prédiction grâce au jeu d'entraînement (training set).
Ensuite nous allons tester le modèle avec le jeu de test (test set) et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.write("")
st.dataframe(df_performance_metrics)