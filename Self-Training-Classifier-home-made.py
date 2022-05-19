from traceback import clear_frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv("spam_clean.csv")  


test_ind = round(len(df)*0.25)
train_ind = test_ind + round(len(df)*0.1)
unlabeled_ind = train_ind + round(len(df)*0.65)


# Partition the data

test = df.iloc[:test_ind]
train = df.iloc[test_ind:train_ind]
unlabeled = df.iloc[train_ind:unlabeled_ind]

X_train = train['text_clean'].values.astype('U')
y_train = train['label_num']

X_unlabeled = unlabeled['text_clean'].values.astype('U')

X_test = test['text_clean'].values.astype('U')
y_test = test['label_num']

tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(X_train).toarray()
X_unlabeled = tfidf.transform(X_unlabeled).toarray()
X_test = tfidf.transform(X_test).toarray()

# Initiate iteration counter
iterations = 0

# Containers to hold f1_scores and # of pseudo-labels
train_f1s = []
test_f1s = []
pseudo_labels = []

# Assign value to initiate while loop
high_prob = True

# Loop will run until there are no more high-probability pseudo-labels
while high_prob==True:
        
    # Fit classifier and make train/test predictions
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

   
    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")
    high_prob=False
    pred_probs = clf.predict_proba(X_unlabeled)
    X_train_tmp=X_train
    preds = clf.predict(X_unlabeled)
    p=0
    for i in range(0,len(pred_probs)) :
        if (pred_probs[i][0]>0.98 or pred_probs[i][1]>0.98):
            high_prob=True
            X_train=np.concatenate((X_train,[X_unlabeled[i]]))
            y_train=np.concatenate((y_train,[preds[i]]))
        else:
            if p==0:
                X_unlabeled_tmp=np.array([X_unlabeled[i]])
                p=1
            else:
                X_unlabeled_tmp=np.concatenate((X_unlabeled_tmp,[X_unlabeled[i]]))

                 
    X_unlabeled=X_unlabeled_tmp
    
    # Update iteration counter
    iterations += 1

y_prediction = clf.predict(X_test)


# Print scores 
accuracy_sc = accuracy_score(y_test, y_prediction)
precision_sc= precision_score(y_test, y_prediction, average='macro')
recall_sc = recall_score(y_test, y_prediction, average='macro')
f1_sc =  f1_score(y_test, y_prediction, average='macro')

performance_metrics=[] 
performance_metrics.append(["Self-training Classifier",accuracy_sc,precision_sc,recall_sc,f1_sc])
print(pd.DataFrame(performance_metrics,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"]))