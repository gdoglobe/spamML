import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Get the mail dataset 
dataset = pd.read_csv("spam_clean.csv")
x = dataset['text_clean'].values.astype('U')
y = dataset['label_num']

# Divided the data into two groups train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# Different supervised algorithms that we used
models = {
    'Logistic_Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'KNN': KNeighborsClassifier()
}

# Different supervised algorithms that we used
params = {
    'Logistic_Regression': {'max_iter':[500],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] },
    'KNN': { 'n_neighbors': [2,3,4,5,6,7,8,9,10], 'weights': ['uniform', 'distance']}
}


def ML_modeling(models, params, X_train, X_test, y_train, y_test):    
    # Check error   
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')
    
    # Store the metrics of the supervised algorithms
    performance_metrics=[]
    # Apply the different algorithms with all parameters
    for key in models.keys():
        model = models[key]
        param = params[key]
        # Check the best parameters to use
        ModelGS = GridSearchCV(model, param, cv=10, error_score=0, refit=True)
        # Create the model with train data
        ModelGS.fit(X_train, y_train)
        # Test the model with test data
        y_prediction = ModelGS.predict(X_test)
        
        # Print scores for the classifier
        accuracy_sc = accuracy_score(y_test, y_prediction)
        precision_sc= precision_score(y_test, y_prediction, average='macro')
        recall_sc = recall_score(y_test, y_prediction, average='macro')
        f1_sc =  f1_score(y_test, y_prediction, average='macro')

        # Add the scores in performance_metrics array 
        performance_metrics.append([key,accuracy_sc,precision_sc,recall_sc,f1_sc])
    # Return performance_metrics array 
    return pd.DataFrame(performance_metrics,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])

# Transform text data on numeric vectors
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)
# Call the main function
df_performance_metrics=ML_modeling(models, params, X_train_tfidf, X_test_tfidf, y_train, y_test)
print(df_performance_metrics)

file = open('metrics_supervised.pkl','wb')
pickle.dump(df_performance_metrics,file)
file.close()