import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import pickle

# Get the mail dataset 
dataset = pd.read_csv("spam_clean.csv")
x = dataset['text_clean'].values.astype('U')
y = dataset['label_num']

# Divided the data into two groups train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# Different unsupervised algorithms that we used
models = {
    'K-means': KMeans(),
    'MB K-Means':MiniBatchKMeans(),
    'AC':AgglomerativeClustering()
}

# Parameters of the unsupervised algorithms
params = {
    'K-means': { 'n_clusters': [2], 'init' :['k-means++','random'],'algorithm':['auto', 'full', 'elkan'],'random_state':[0,1,2,3,4]},
    'MB K-Means': { 'n_clusters':[2],'init' :['k-means++','random'],'random_state':[0,1,2,3,4]},
    'AC': {},    
}


def ML_modeling(models, params, X_train, X_test, y_test):    
    
    # Check error
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')

    # Store the metrics of the unsupervised algorithms
    performance_metrics=[]
    # Store the reverse forecast to know wich is the spams and the hams
    y_prediction2=[]
    # Apply the different algorithms with all parameters
    for key in models.keys():
        model = models[key]
        param = params[key]
        # Apply AgglomerativeClustering independantly
        if(str(model)=='AgglomerativeClustering()'):
            X_train=X_train.toarray()
            X_test=X_test.toarray()
            # Create the model with train data
            modelAC = AgglomerativeClustering(n_clusters=2).fit(X_train)
            # Test the model with test data
            y_prediction=modelAC.fit_predict(X_test)
        # Apply the other algorithms
        else:
            # Check the best parameters to use
            modelGS = GridSearchCV(model, param, cv=10, error_score=0, refit=True) 
            # Create the model with train data
            modelGS.fit(X_train)
            # Test the model with test data
            y_prediction = modelGS.predict(X_test)
        
        # fill the reverse result in y_prediction2 array
        for i in y_prediction:
            if i==0:
                y_prediction2.append(1)
            else:
                y_prediction2.append(0) 

        # Print scores for the classifier
        accuracy_sc = accuracy_score(y_test, y_prediction)
        precision_sc= precision_score(y_test, y_prediction, average='macro')
        recall_sc = recall_score(y_test, y_prediction, average='macro')
        f1_sc =  f1_score(y_test, y_prediction, average='macro')

        # Print scores for the classifier
        accuracy_sc_2 = accuracy_score(y_test, y_prediction2)
        precision_sc_2= precision_score(y_test, y_prediction2, average='macro')
        recall_sc_2 = recall_score(y_test, y_prediction2, average='macro')
        f1_sc_2 =  f1_score(y_test, y_prediction2, average='macro')

        # Add the scores in performance_metrics array      
        performance_metrics.append([key,accuracy_sc,precision_sc,recall_sc,f1_sc])
        performance_metrics.append([key,accuracy_sc_2,precision_sc_2,recall_sc_2,f1_sc_2])
        # Empty y_prediction2 array
        y_prediction2=[]
    # return performance_metrics array 
    return pd.DataFrame(performance_metrics,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])

# Transform text data on numeric vectors
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)
# Call the main function
df_performance_metrics=ML_modeling(models, params, X_train_tfidf, X_test_tfidf, y_test)
