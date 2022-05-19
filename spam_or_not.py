import pickle
from PIL import Image
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
tfidf = TfidfVectorizer(stop_words='english')
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

# recovery of previously registered models
file = open('pkl/Models_S.pkl', 'rb')
table= pickle.load(file)
file.close()
modelLR=table[0]
modelNB=table[1]
modelKNN=table[2]
tfidf_S=table[3]
file = open('pkl/Models_SS1.pkl', 'rb')
table2= pickle.load(file)
file.close()
model_SF_LR=table2[0]
model_SF_NB=table2[1]
tfidf_SS1=table2[2]
tableau_text=[]


# pre-processing text function 
def clean_text(text):
    new_text=text.lower()
    clean_text= re.sub("[^a-z]+"," ",new_text)
    clean_text_stopwords = ""
    for i in clean_text.split(" ")[1:]:
        if not i in stopwords and len(i) > 3:
            wordlem=lemmatizer.lemmatize(i)
            wordStem=stemmer.stem(wordlem)
            clean_text_stopwords += wordStem
            clean_text_stopwords += " "
    return clean_text_stopwords


# Display sidebare
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>MAIL MENU</h1>", unsafe_allow_html=True)
mail_text = st.sidebar.text_area("Entrez le contenu texte de votre mail :")
option = st.sidebar.selectbox(
     'Quel algorithme choisissez-vous ?',
     ('Logistic Regression', 'Naive Bayes', 'KNN','SelfTrainingClassifierLR','SelfTrainingClassifierNB'))
button = st.sidebar.button("Lancer")


# Display main page
st.markdown("<h1 style='text-align: center; color: red;'>SPAM OR NOT ?</h1>", unsafe_allow_html=True)

# if the button has been pressed 
if button:
     # display of the entered mail
     st.write("Voici, le mail que vous cherchez à examiner : " + mail_text)
     tableau_text.append(clean_text(mail_text))
     # model choice
     if(option=='Logistic Regression'):
           st.write("Pour déterminer la nature du mail, vous avez choisi l'algorithme Logistic Regression (LR).")
           st.write("D'après Logistic Regression, votre mail est :")
           predicted= modelLR.predict(tfidf_S.transform(tableau_text))
           resultat=predicted[0]
     elif(option=='Naive Bayes'):
           st.write("Pour déterminer la nature du mail, vous avez choisi la classification naïve bayésienne(NB).")
           st.write("D'après NB, votre mail est :")
           predicted= modelNB.predict(tfidf_S.transform(tableau_text))
           resultat=predicted[0]
     elif(option=='KNN'): 
           st.write("Pour déterminer la nature du mail, vous avez choisi l'algorithme des K plus proches voisins (KNN).")
           st.write("D'après KNN, votre mail est :")
           predicted= modelKNN.predict(tfidf_S.transform(tableau_text))
           resultat=predicted[0]
     elif(option=='SelfTrainingClassifierLR'): 
           st.write("Pour déterminer la nature du mail, vous avez choisi SelfTrainingClassifier avec Logistic Regression.")
           st.write("D'après SelfTrainingClassifier, votre mail est :")
           predicted= model_SF_LR.predict(tfidf_SS1.transform(tableau_text).toarray())
           resultat=predicted[0]
     elif(option=='SelfTrainingClassifierNB'): 
           st.write("Pour déterminer la nature du mail, vous avez choisi SelfTrainingClassifier avec Naive Bayes.")
           st.write("D'après SelfTrainingClassifier, votre mail est :")
           predicted= model_SF_NB.predict(tfidf_SS1.transform(tableau_text).toarray())
           resultat=predicted[0]    
     # display of the result
     if(resultat==0):
         image = Image.open('picture/notspam.png')
         col1, col2, col3 = st.columns([0.6, 1, 0.4])
         col2.image([image])
     elif(resultat==1):
         image = Image.open('picture/spam.png')
         col1, col2, col3 = st.columns([0.6, 1, 0.4])
         col2.image([image])
     tableau_text=[]
