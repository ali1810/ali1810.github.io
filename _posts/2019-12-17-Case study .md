

```python
# import data.
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np
import re
df= pd.read_csv('/Users/ali/review.csv',header=0, encoding='latin1')
df['review'] = df['text']
# create new colums for each season if months match conditions.
df['winter'] = df['review_date'].apply(lambda x : 'winter' if re.findall(r'^\d*', x)[0] in ['12','1','2'] else np.nan)
df['summer'] = df['review_date'].apply(lambda x : 'summer' if re.findall(r'^\d*', x)[0] in ['6','7','8'] else np.nan)
df['fall'] = df['review_date'].apply(lambda x : 'fall' if re.findall(r'^\d*', x)[0] in ['9','10','11'] else np.nan)
df['spring'] = df['review_date'].apply(lambda x : 'spring' if re.findall(r'^\d*', x)[0] in ['3','4','5'] else np.nan)
```

# Topics Classification.



```python
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns
import numpy as np
```

# Topic Classification without Gridsearch

## Multinomial Naive Bayes


```python
# define a function to evaluate NB model.
def NB_topic_classification(train_file_name):
    # import dataset.
    categ_data = pd.read_csv(train_file_name , header = 0)    
    classes = categ_data.label.values
    classes = [i.split(",") for i in classes]
    # one hot encoding
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(classes)
    
    # split dataset into traning and testing.
    X_train, X_test, Y_train, Y_test = train_test_split(\
                    categ_data.review, Y, test_size=0.3, random_state=0)
    
    # create a pipeline to vectorize and classify.
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english",\
                                  min_df=2)),
        ('clf', OneVsRestClassifier(MultinomialNB(alpha = 0.5 )))])
    
    # train model with training part of the datastet.
    classifier.fit(X_train, Y_train)
    
    # predict testing data and store them
    predicted = classifier.predict(X_test)

    predicted.shape
    # compare predicted with the ground truth and report performance.
    print(classification_report\
          (Y_test, predicted, target_names=mlb.classes_))
```


```python
NB_topic_classification("label restaurant data.csv" )
```

    precision    recall  f1-score   support
    
                   ambience       0.95      0.29      0.44       140
    anecdotes/miscellaneous       0.74      0.56      0.64       342
                       food       0.83      0.72      0.77       366
                      price       0.93      0.36      0.52       102
                    service       0.86      0.50      0.64       173
    
                  micro avg       0.82      0.55      0.66      1123
                  macro avg       0.86      0.49      0.60      1123
               weighted avg       0.83      0.55      0.65      1123
                samples avg       0.62      0.58      0.59      1123
    
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels.
      'precision', 'predicted', average, warn_for)
    

## Support Vector Machine


```python
# This function is to evaluate SVM model.
# The code is referenced from the NB model eveluation above;
# however, the model has changed to SVM in the pipeline definition.

def SVM_topic_classification(train_file_name):
    
    categ_data = pd.read_csv(train_file_name , header = 0)    
    classes = categ_data.label.values
    classes = [i.split(",") for i in classes]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(classes)

    X_train, X_test, Y_train, Y_test = train_test_split(\
                    categ_data.review, Y, test_size=0.3, random_state=0)

    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english",\
                                  min_df=2)),
        ('clf', OneVsRestClassifier(LinearSVC(C = 1)))])

    classifier.fit(X_train, Y_train)

    predicted = classifier.predict(X_test)

    predicted.shape
    
    print(classification_report\
          (Y_test, predicted, target_names=mlb.classes_))
```


```python
SVM_topic_classification("label restaurant data.csv")
```

    precision    recall  f1-score   support
    
                   ambience       0.87      0.54      0.66       140
    anecdotes/miscellaneous       0.71      0.73      0.72       342
                       food       0.85      0.80      0.82       366
                      price       0.88      0.74      0.80       102
                    service       0.85      0.70      0.77       173
    
                  micro avg       0.80      0.73      0.76      1123
                  macro avg       0.83      0.70      0.76      1123
               weighted avg       0.81      0.73      0.76      1123
                samples avg       0.74      0.73      0.72      1123
    
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels.
      'precision', 'predicted', average, warn_for)
    

## K-nearest neighbour


```python
# This function is to evaluate KNN model.
# The code is referenced from the NB model eveluation above;
# however, the model has changed to KNN in the pipeline definition.


def KNN_topic_classification(train_file_name):
    
    categ_data = pd.read_csv(train_file_name , header = 0)    
    classes = categ_data.label.values
    classes = [i.split(",") for i in classes]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(classes)

    X_train, X_test, Y_train, Y_test = train_test_split(\
                    categ_data.review, Y, test_size=0.3, random_state=0)

    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english",\
                                  min_df=2)),
        ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1)))])

    classifier.fit(X_train, Y_train)

    predicted = classifier.predict(X_test)

    predicted.shape
    
    print(classification_report\
          (Y_test, predicted, target_names=mlb.classes_))  

```


```python
KNN_topic_classification("label restaurant data.csv" )
```

    precision    recall  f1-score   support
    
                   ambience       0.67      0.11      0.20       140
    anecdotes/miscellaneous       0.42      0.87      0.57       342
                       food       0.73      0.25      0.37       366
                      price       0.75      0.29      0.42       102
                    service       0.56      0.23      0.33       173
    
                  micro avg       0.49      0.42      0.46      1123
                  macro avg       0.63      0.35      0.38      1123
               weighted avg       0.60      0.42      0.41      1123
                samples avg       0.49      0.46      0.46      1123
    
    

# Topic Classification with Gridsearch


```python
categ_data = pd.read_csv("label restaurant data.csv" , header = 0)   
categ_data.head()
classes = categ_data.label.values
classes = [i.split(",") for i in classes]

```


```python
# implement one hot encoding.
mlb = MultiLabelBinarizer()
Y=mlb.fit_transform(classes)

classes_label = mlb.classes_


ambience = Y[:,0]
food = Y[:,2]
miscellaneous = Y[:,1]
service = Y[:,4]
price = Y[:,3]

```

## Topic Classification using Multinomial Naive Bayes.


```python

def NB_topic_classification(review_data, feature):
    

    X_train, X_test, y_train, y_test = train_test_split(review_data, feature , \
                                                        test_size=0.25, random_state=0)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

    parameters = {'tfidf__min_df':[1, 2, 3],
                  'tfidf__max_df': [0.995 , 0.999, 1.0],
                  'tfidf__stop_words':[None,"english"],
                  'clf__alpha': [0.5, 1.0, 2.0 , 5.0]}

    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)


    gs_clf = gs_clf.fit(X_train, y_train)

    for param_name in gs_clf.best_params_:
        print("{} : {}".format(param_name , gs_clf.best_params_[param_name]))
    print("best f1 score:", gs_clf.best_score_)

    clf_alpha = gs_clf.best_params_["clf__alpha"]
    tfidf_min_df = gs_clf.best_params_["tfidf__min_df"]
    tfidf_max_df = gs_clf.best_params_["tfidf__max_df"]
    tfidf_stop_words = gs_clf.best_params_["tfidf__stop_words"]

    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=tfidf_stop_words,\
                                  min_df=tfidf_min_df, max_df = tfidf_max_df)),
        ('clf', MultinomialNB(alpha = clf_alpha ))])

    clf = classifier.fit(X_train,  y_train)

    labels=sorted(np.unique(feature))
    labels = list(map(str, labels))

    predicted = classifier.predict(X_test)
    
    return predicted


```


```python
predicted_ambience =  NB_topic_classification(categ_data.review, ambience)
predicted_miscellaneous=  NB_topic_classification(categ_data.review, miscellaneous)
predicted_food =  NB_topic_classification(categ_data.review, food)
predicted_price =  NB_topic_classification(categ_data.review, price)
predicted_service =  NB_topic_classification(categ_data.review, service)
```

    ification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    clf__alpha : 0.5
    tfidf__max_df : 0.995
    tfidf__min_df : 3
    tfidf__stop_words : english
    best f1 score: 0.668024906667693
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    clf__alpha : 0.5
    tfidf__max_df : 0.995
    tfidf__min_df : 3
    tfidf__stop_words : english
    best f1 score: 0.7959727136933766
    


```python
X_train, X_test, y_train, y_test = train_test_split(categ_data.review, Y , \
                                                    test_size=0.25, random_state=0)

zip_all = list(zip(predicted_ambience, predicted_miscellaneous, predicted_food, predicted_price, predicted_service))

print(classification_report\
      (y_test, np.array(list(zip_all)), target_names=mlb.classes_))
```

    precision    recall  f1-score   support
    
                   ambience       0.95      0.30      0.45       121
    anecdotes/miscellaneous       0.78      0.56      0.65       289
                       food       0.86      0.70      0.77       301
                      price       0.97      0.41      0.58        80
                    service       0.88      0.56      0.68       145
    
                  micro avg       0.85      0.56      0.67       936
                  macro avg       0.89      0.50      0.63       936
               weighted avg       0.86      0.56      0.66       936
                samples avg       0.62      0.59      0.60       936
    
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels.
      'precision', 'predicted', average, warn_for)
    

## Topic Classification using Support Vector Machine


```python

def SVM_topic_classification(review_data, feature):
    

    X_train, X_test, y_train, y_test = train_test_split(review_data, feature , \
                                                        test_size=0.25, random_state=0)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', svm.SVC())])

    parameters = {'tfidf__min_df':[1, 2, 3],
                  'tfidf__max_df': [0.995 , 0.999 , 1.0],
                  'tfidf__stop_words':[None,"english"],
                  'clf__C':[ 1, 2, 5],
                 'clf__kernel':['linear']}

    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)


    gs_clf = gs_clf.fit(X_train, y_train)

    for param_name in gs_clf.best_params_:
        print("{} : {}".format(param_name , gs_clf.best_params_[param_name]))
    print("best f1 score:", gs_clf.best_score_)

    clf_C = gs_clf.best_params_["clf__C"]
    clf_kernel = gs_clf.best_params_["clf__kernel"]
    tfidf_min_df = gs_clf.best_params_["tfidf__min_df"]
    tfidf_max_df = gs_clf.best_params_["tfidf__max_df"]
    tfidf_stop_words = gs_clf.best_params_["tfidf__stop_words"]

    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=tfidf_stop_words,\
                                  min_df=tfidf_min_df, max_df = tfidf_max_df)),
        ('clf', svm.SVC(C = clf_C, kernel = clf_kernel))])

    clf = classifier.fit(X_train,  y_train)

    labels=sorted(np.unique(feature))
    labels = list(map(str, labels))

    predicted = classifier.predict(X_test)
    
    return predicted

```


```python
predicted_ambience =  SVM_topic_classification(categ_data.review, ambience)
predicted_miscellaneous=  SVM_topic_classification(categ_data.review, miscellaneous)
predicted_food =  SVM_topic_classification(categ_data.review, food)
predicted_price =  SVM_topic_classification(categ_data.review, price)
predicted_service =  SVM_topic_classification(categ_data.review, service)
```

    clf__C : 2
    clf__kernel : linear
    tfidf__max_df : 0.995
    tfidf__min_df : 1
    tfidf__stop_words : english
    best f1 score: 0.8115904458156638
    clf__C : 1
    clf__kernel : linear
    tfidf__max_df : 0.995
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.7991135102581619
    clf__C : 1
    clf__kernel : linear
    tfidf__max_df : 0.995
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.8666034900184318
    clf__C : 2
    clf__kernel : linear
    tfidf__max_df : 0.995
    tfidf__min_df : 1
    tfidf__stop_words : english
    best f1 score: 0.8534716551437846
    clf__C : 2
    clf__kernel : linear
    tfidf__max_df : 0.995
    tfidf__min_df : 1
    tfidf__stop_words : english
    best f1 score: 0.8787895157563381
    


```python
X_train, X_test, y_train, y_test = train_test_split(categ_data.review, Y , \
                                                    test_size=0.25, random_state=0)

zip_all = list(zip(predicted_ambience, predicted_miscellaneous, predicted_food, predicted_price, predicted_service))

print(classification_report\
      (y_test, np.array(list(zip_all)), target_names=mlb.classes_))
```

    precision    recall  f1-score   support
    
                   ambience       0.95      0.30      0.45       121
    anecdotes/miscellaneous       0.78      0.56      0.65       289
                       food       0.86      0.70      0.77       301
                      price       0.97      0.41      0.58        80
                    service       0.88      0.56      0.68       145
    
                  micro avg       0.85      0.56      0.67       936
                  macro avg       0.89      0.50      0.63       936
               weighted avg       0.86      0.56      0.66       936
                samples avg       0.62      0.59      0.60       936
    
    

## Topic Classification using K-Nearest Neighbour


```python

def KNN_topic_classification(review_data, feature):
    

    X_train, X_test, y_train, y_test = train_test_split(review_data, feature , \
                                                        test_size=0.25, random_state=0)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier())])
    
    k_range = list(range(1, 10))

    parameters = {'tfidf__min_df':[1, 2, 3],
                  'tfidf__stop_words':[None,"english"],
                  'clf__n_neighbors': k_range}

    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)


    gs_clf = gs_clf.fit(X_train, y_train)

    for param_name in gs_clf.best_params_:
        print("{} : {}".format(param_name , gs_clf.best_params_[param_name]))
    print("best f1 score:", gs_clf.best_score_)

    clf_k = gs_clf.best_params_["clf__n_neighbors"]
    tfidf_min_df = gs_clf.best_params_["tfidf__min_df"]
    tfidf_stop_words = gs_clf.best_params_["tfidf__stop_words"]

    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=tfidf_stop_words,\
                                  min_df=tfidf_min_df)),
        ('clf', KNeighborsClassifier(n_neighbors= clf_k))])

    clf = classifier.fit(X_train,  y_train)

    labels=sorted(np.unique(feature))
    labels = list(map(str, labels))

    predicted = classifier.predict(X_test)
    
    return predicted
```


```python
predicted_ambience =  KNN_topic_classification(categ_data.review, ambience)
predicted_miscellaneous= KNN_topic_classification(categ_data.review, miscellaneous)
predicted_food =  KNN_topic_classification(categ_data.review, food)
predicted_price =  KNN_topic_classification(categ_data.review, price)
predicted_service =  KNN_topic_classification(categ_data.review, service)
```

    cation.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    clf__n_neighbors : 1
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.6425241878899145
    clf__n_neighbors : 7
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.7457548560703786
    clf__n_neighbors : 9
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.787822939909712
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    clf__n_neighbors : 3
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.7020501053173683
    /Users/anishumesh/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    clf__n_neighbors : 9
    tfidf__min_df : 1
    tfidf__stop_words : None
    best f1 score: 0.755653894590898
    


```python
X_train, X_test, y_train, y_test = train_test_split(categ_data.review, Y , \
                                                    test_size=0.25, random_state=0)

zip_all = list(zip(predicted_ambience, predicted_miscellaneous, predicted_food, predicted_price, predicted_service))

print(classification_report\
      (y_test, np.array(list(zip_all)), target_names=mlb.classes_))
```

    precision    recall  f1-score   support
    
                   ambience       0.95      0.30      0.45       121
    anecdotes/miscellaneous       0.78      0.56      0.65       289
                       food       0.86      0.70      0.77       301
                      price       0.97      0.41      0.58        80
                    service       0.88      0.56      0.68       145
    
                  micro avg       0.85      0.56      0.67       936
                  macro avg       0.89      0.50      0.63       936
               weighted avg       0.86      0.56      0.66       936
                samples avg       0.62      0.59      0.60       936
    
    

## Feature classification using SVM (Best Performance)


```python
All_review_to_classify = pd.read_csv("/Users/anishumesh/food_reviews.csv")
All_review_to_classify['review'] = All_review_to_classify['text']
All_review_to_classify = All_review_to_classify.review

def feature_classification(reviews):
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', svm.SVC())])

    labels=sorted(np.unique(ambience))
    labels = list(map(str, labels))


    ambience_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words= "english", min_df=1, max_df = 0.995)),
        ('clf', svm.SVC(C = 2, kernel = 'linear'))])

    ambience_clf = ambience_classifier.fit(categ_data.review, ambience)

    ambience_predicted = ambience_classifier.predict(reviews)


    miscel_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words= None, min_df=1, max_df = 0.995)),
        ('clf', svm.SVC(C = 1, kernel = 'linear'))])

    miscel_clf = miscel_classifier.fit(categ_data.review, miscellaneous)

    miscel_predicted = miscel_classifier.predict(reviews)


    food_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words= None, min_df=1, max_df = 0.995)),
        ('clf', svm.SVC(C = 1, kernel = 'linear'))])

    food_clf = food_classifier.fit(categ_data.review, food)

    food_predicted = food_classifier.predict(reviews)


    price_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words= "english", min_df=1, max_df = 0.995)),
        ('clf', svm.SVC(C = 2, kernel = 'linear'))])

    price_clf = price_classifier.fit(categ_data.review, price)

    price_predicted = price_classifier.predict(reviews)


    service_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words= "english", min_df=1, max_df = 0.995)),
        ('clf', svm.SVC(C = 2, kernel = 'linear'))])

    service_clf = service_classifier.fit(categ_data.review, service)

    service_predicted = service_classifier.predict(reviews)
    
    combined_classes = np.array(list(zip(ambience_predicted, miscel_predicted, \
                                     food_predicted, price_predicted, service_predicted)))
    return combined_classes


```


```python
combined_classes = feature_classification(All_review_to_classify)

count_classes = np.sum(combined_classes , axis=0)


fig1, ax1 = plt.subplots()
ax1.pie(count_classes, explode= (0, 0, 0.1, 0 , 0) , labels = mlb.classes_ ,autopct='%1.1f%%')
ax1.axis('equal')

```




    (-1.1861625405042877,
     1.1041029995557592,
     -1.0648191950982024,
     1.1631035656679731)




![png](casestudy_files/casestudy_30_1.png)



```python
combined_classes
```




    array([[0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 0]])




```python
classify
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ambience</th>
      <th>Miscelleneous</th>
      <th>Food</th>
      <th>Price</th>
      <th>Service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

classify = pd.DataFrame(list(map(np.ravel, combined_classes)))
classify.columns = ['Ambience', 'Miscelleneous', 'Food', 'Price', 'Service']
classify
All_Reviews = pd.DataFrame(All_review_to_classify)
Final_Reviews = All_Reviews.join(classify)
Final_Reviews['Food Mentioned'] = df['menu_item']
Final_Reviews['Rating'] = df['review_rating']
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sentiment = Final_Reviews['review'].apply(lambda x: analyser.polarity_scores(x))
Final_Reviews = pd.concat([Final_Reviews,sentiment.apply(pd.Series)],1)
Final_Reviews
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>Ambience</th>
      <th>Miscelleneous</th>
      <th>Food</th>
      <th>Price</th>
      <th>Service</th>
      <th>Food Mentioned</th>
      <th>Rating</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I always order the most basic burger whenever ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>The Greek Burger,The Mexican Burger,Parmesan T...</td>
      <td>5.0</td>
      <td>0.019</td>
      <td>0.709</td>
      <td>0.271</td>
      <td>0.9847</td>
    </tr>
    <tr>
      <th>1</th>
      <td>After my second time here  I can truly say I r...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mussels and Clams,Cavatelli alla Norcina,Musse...</td>
      <td>5.0</td>
      <td>0.019</td>
      <td>0.668</td>
      <td>0.312</td>
      <td>0.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all  their minimum for delivery is on...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Tea,Fries,Fries,Fries,Cilantro Lime Salad,Frie...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.836</td>
      <td>0.164</td>
      <td>0.9827</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Was not disappointed once I was in range and c...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Dinner Plate,Wings,Wings,Brunch Special,Wings,...</td>
      <td>5.0</td>
      <td>0.056</td>
      <td>0.771</td>
      <td>0.173</td>
      <td>0.9352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Had Croque Monsieur  it can come in ham  turke...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>30 Crepe Xpress,French Onion Soup,Water,15 Cro...</td>
      <td>5.0</td>
      <td>0.023</td>
      <td>0.851</td>
      <td>0.126</td>
      <td>0.9072</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Delish  I used to love this cute French crepes...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>La Colombienne,La Larzac,La Scandinave,La Bres...</td>
      <td>4.0</td>
      <td>0.027</td>
      <td>0.764</td>
      <td>0.209</td>
      <td>0.9867</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Had to "do the dirty" while in SoCal   I'm veg...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>The Dirty Corn,The Green Dog,The Green Dog,The...</td>
      <td>4.0</td>
      <td>0.082</td>
      <td>0.693</td>
      <td>0.225</td>
      <td>0.9401</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I finally got to try this sandwich shop  They ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Cannoli,Macaroni Salad,Cannoli,Cannoli,Chicken...</td>
      <td>3.0</td>
      <td>0.029</td>
      <td>0.764</td>
      <td>0.207</td>
      <td>0.9800</td>
    </tr>
    <tr>
      <th>8</th>
      <td>My favorite tacos ever  Everyone I take here I...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Steak Picado,Chicarron,Mole Poblano,Quesadilla...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.800</td>
      <td>0.200</td>
      <td>0.9833</td>
    </tr>
    <tr>
      <th>9</th>
      <td>After a fulfilling day at the LA Flower Distri...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Chorizo,Quesadilla,Steak Picado,Quesadilla,Hor...</td>
      <td>5.0</td>
      <td>0.035</td>
      <td>0.790</td>
      <td>0.175</td>
      <td>0.9855</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Holy moly I just got my socks blown off  That ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Skinny Jimmy,Half Bird,Dark,Wings,Batter's Box...</td>
      <td>5.0</td>
      <td>0.020</td>
      <td>0.873</td>
      <td>0.106</td>
      <td>0.9891</td>
    </tr>
    <tr>
      <th>11</th>
      <td>These sandwiches are better than Cole's  There...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Egg,Avocado,Grilled Chicken Sandwich,Breakfast...</td>
      <td>5.0</td>
      <td>0.062</td>
      <td>0.720</td>
      <td>0.218</td>
      <td>0.9771</td>
    </tr>
    <tr>
      <th>12</th>
      <td>This is an update from last year's post   I we...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Seafood,Beef,Dessert,Seafood,Wagyu Beef,Beef,S...</td>
      <td>5.0</td>
      <td>0.091</td>
      <td>0.801</td>
      <td>0.108</td>
      <td>0.5771</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Street parking only  Very hard to find Paid Lo...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Lobster,Lobster,Yellowtail,Toro,Blue Crab,Bay ...</td>
      <td>5.0</td>
      <td>0.021</td>
      <td>0.774</td>
      <td>0.206</td>
      <td>0.9826</td>
    </tr>
    <tr>
      <th>14</th>
      <td>This place is a hidden gem in the LA area   Th...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Las Jamburgesas,Chancluditas (2 Pcs)</td>
      <td>5.0</td>
      <td>0.018</td>
      <td>0.771</td>
      <td>0.212</td>
      <td>0.9956</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Food  The food here is very good  The past...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Apple Sauce,Cole Slaw,Chili,Pastrami,Chili Che...</td>
      <td>5.0</td>
      <td>0.043</td>
      <td>0.795</td>
      <td>0.162</td>
      <td>0.9824</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Visiting the LA area for a conference  I gave ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Egg and Cheese,Cobb Salad,Anti - Flu Juice,Ita...</td>
      <td>5.0</td>
      <td>0.018</td>
      <td>0.618</td>
      <td>0.364</td>
      <td>0.9960</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3 5 stars Parking  street parkingFood  ordered...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Chilaquiles,Huevos Rancheros,Enchiladas de Mol...</td>
      <td>3.0</td>
      <td>0.035</td>
      <td>0.808</td>
      <td>0.157</td>
      <td>0.6529</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Came here with my family after many years crav...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Chicken,Beef,Supreme,Pepperoni,Chicken,Beef,Su...</td>
      <td>5.0</td>
      <td>0.007</td>
      <td>0.821</td>
      <td>0.172</td>
      <td>0.9916</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The lemon herb chicken sandwich   What is in i...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Lemon Herb Chicken Sandwich,Tuna Sandwich,Lemo...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.910</td>
      <td>0.090</td>
      <td>0.7217</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Best sushi and sashimi near Los Angeles Price ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Sashimi,Rice,California Roll,Sashimi,Rice,Soup...</td>
      <td>4.0</td>
      <td>0.000</td>
      <td>0.766</td>
      <td>0.234</td>
      <td>0.7469</td>
    </tr>
    <tr>
      <th>21</th>
      <td>This review is long overdue  Hands down the be...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Simple</td>
      <td>5.0</td>
      <td>0.040</td>
      <td>0.764</td>
      <td>0.196</td>
      <td>0.9830</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Was across the way at the coffee shop and want...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Texas BBQ Wrap,The Stack Combo,Mac N Cheese,BB...</td>
      <td>4.0</td>
      <td>0.033</td>
      <td>0.847</td>
      <td>0.121</td>
      <td>0.9355</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Top notch cafe inside of the cool Sci Arc camp...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Turkey Burger,Ghost Burger,Ghost Burger,Black ...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.760</td>
      <td>0.240</td>
      <td>0.9735</td>
    </tr>
    <tr>
      <th>24</th>
      <td>I had Semi booked because the brunch pictures ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Tomato Bisque,Tomato Bisque,Nitro Cold Brew,Ni...</td>
      <td>4.0</td>
      <td>0.025</td>
      <td>0.767</td>
      <td>0.208</td>
      <td>0.9972</td>
    </tr>
  </tbody>
</table>
</div>




```python
filter = (Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound'] >= 7.5) & (Final_Reviews['Rating']>4)
Final_Reviews.loc[filter,"Revisit Intention"] = 'Definitely'

filter = ((Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound'] >= 6.5 )& (Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound']<7.5)) & (Final_Reviews['Rating']>4)
Final_Reviews.loc[filter,"Revisit Intention"] = 'Probably'

filter = ((Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound'] >= 4.5 )& (Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound']<6.5)) & (Final_Reviews['Rating']>=3)
Final_Reviews.loc[filter,"Revisit Intention"] = 'May or May Not'

filter = ((Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound'] >= 3.5 )& (Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound']<4.5)) & (Final_Reviews['Rating']>=3)
Final_Reviews.loc[filter,"Revisit Intention"] = 'Probably Not'

filter = ((Final_Reviews['Ambience'] + Final_Reviews['Miscelleneous'] + Final_Reviews['Food'] + Final_Reviews['Price'] + Final_Reviews['Service'] + Final_Reviews['Rating'] + Final_Reviews['compound']<3.5)) & (Final_Reviews['Rating']<3)
Final_Reviews.loc[filter,"Revisit Intention"] = 'Definitely Not'
Final_Reviews
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>Ambience</th>
      <th>Miscelleneous</th>
      <th>Food</th>
      <th>Price</th>
      <th>Service</th>
      <th>Food Mentioned</th>
      <th>Rating</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>compound</th>
      <th>Revisit Intention</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I always order the most basic burger whenever ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>The Greek Burger,The Mexican Burger,Parmesan T...</td>
      <td>5.0</td>
      <td>0.019</td>
      <td>0.709</td>
      <td>0.271</td>
      <td>0.9847</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>1</th>
      <td>After my second time here  I can truly say I r...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mussels and Clams,Cavatelli alla Norcina,Musse...</td>
      <td>5.0</td>
      <td>0.019</td>
      <td>0.668</td>
      <td>0.312</td>
      <td>0.9994</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all  their minimum for delivery is on...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Tea,Fries,Fries,Fries,Cilantro Lime Salad,Frie...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.836</td>
      <td>0.164</td>
      <td>0.9827</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Was not disappointed once I was in range and c...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Dinner Plate,Wings,Wings,Brunch Special,Wings,...</td>
      <td>5.0</td>
      <td>0.056</td>
      <td>0.771</td>
      <td>0.173</td>
      <td>0.9352</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Had Croque Monsieur  it can come in ham  turke...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>30 Crepe Xpress,French Onion Soup,Water,15 Cro...</td>
      <td>5.0</td>
      <td>0.023</td>
      <td>0.851</td>
      <td>0.126</td>
      <td>0.9072</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Delish  I used to love this cute French crepes...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>La Colombienne,La Larzac,La Scandinave,La Bres...</td>
      <td>4.0</td>
      <td>0.027</td>
      <td>0.764</td>
      <td>0.209</td>
      <td>0.9867</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Had to "do the dirty" while in SoCal   I'm veg...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>The Dirty Corn,The Green Dog,The Green Dog,The...</td>
      <td>4.0</td>
      <td>0.082</td>
      <td>0.693</td>
      <td>0.225</td>
      <td>0.9401</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I finally got to try this sandwich shop  They ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Cannoli,Macaroni Salad,Cannoli,Cannoli,Chicken...</td>
      <td>3.0</td>
      <td>0.029</td>
      <td>0.764</td>
      <td>0.207</td>
      <td>0.9800</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>8</th>
      <td>My favorite tacos ever  Everyone I take here I...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Steak Picado,Chicarron,Mole Poblano,Quesadilla...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.800</td>
      <td>0.200</td>
      <td>0.9833</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>9</th>
      <td>After a fulfilling day at the LA Flower Distri...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Chorizo,Quesadilla,Steak Picado,Quesadilla,Hor...</td>
      <td>5.0</td>
      <td>0.035</td>
      <td>0.790</td>
      <td>0.175</td>
      <td>0.9855</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Holy moly I just got my socks blown off  That ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Skinny Jimmy,Half Bird,Dark,Wings,Batter's Box...</td>
      <td>5.0</td>
      <td>0.020</td>
      <td>0.873</td>
      <td>0.106</td>
      <td>0.9891</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>11</th>
      <td>These sandwiches are better than Cole's  There...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Egg,Avocado,Grilled Chicken Sandwich,Breakfast...</td>
      <td>5.0</td>
      <td>0.062</td>
      <td>0.720</td>
      <td>0.218</td>
      <td>0.9771</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>12</th>
      <td>This is an update from last year's post   I we...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Seafood,Beef,Dessert,Seafood,Wagyu Beef,Beef,S...</td>
      <td>5.0</td>
      <td>0.091</td>
      <td>0.801</td>
      <td>0.108</td>
      <td>0.5771</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Street parking only  Very hard to find Paid Lo...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Lobster,Lobster,Yellowtail,Toro,Blue Crab,Bay ...</td>
      <td>5.0</td>
      <td>0.021</td>
      <td>0.774</td>
      <td>0.206</td>
      <td>0.9826</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>14</th>
      <td>This place is a hidden gem in the LA area   Th...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Las Jamburgesas,Chancluditas (2 Pcs)</td>
      <td>5.0</td>
      <td>0.018</td>
      <td>0.771</td>
      <td>0.212</td>
      <td>0.9956</td>
      <td>Definitely</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Food  The food here is very good  The past...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Apple Sauce,Cole Slaw,Chili,Pastrami,Chili Che...</td>
      <td>5.0</td>
      <td>0.043</td>
      <td>0.795</td>
      <td>0.162</td>
      <td>0.9824</td>
      <td>Definitely</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Visiting the LA area for a conference  I gave ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Egg and Cheese,Cobb Salad,Anti - Flu Juice,Ita...</td>
      <td>5.0</td>
      <td>0.018</td>
      <td>0.618</td>
      <td>0.364</td>
      <td>0.9960</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3 5 stars Parking  street parkingFood  ordered...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Chilaquiles,Huevos Rancheros,Enchiladas de Mol...</td>
      <td>3.0</td>
      <td>0.035</td>
      <td>0.808</td>
      <td>0.157</td>
      <td>0.6529</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Came here with my family after many years crav...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Chicken,Beef,Supreme,Pepperoni,Chicken,Beef,Su...</td>
      <td>5.0</td>
      <td>0.007</td>
      <td>0.821</td>
      <td>0.172</td>
      <td>0.9916</td>
      <td>Definitely</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The lemon herb chicken sandwich   What is in i...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Lemon Herb Chicken Sandwich,Tuna Sandwich,Lemo...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.910</td>
      <td>0.090</td>
      <td>0.7217</td>
      <td>Probably</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Best sushi and sashimi near Los Angeles Price ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Sashimi,Rice,California Roll,Sashimi,Rice,Soup...</td>
      <td>4.0</td>
      <td>0.000</td>
      <td>0.766</td>
      <td>0.234</td>
      <td>0.7469</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>21</th>
      <td>This review is long overdue  Hands down the be...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Simple</td>
      <td>5.0</td>
      <td>0.040</td>
      <td>0.764</td>
      <td>0.196</td>
      <td>0.9830</td>
      <td>Definitely</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Was across the way at the coffee shop and want...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Texas BBQ Wrap,The Stack Combo,Mac N Cheese,BB...</td>
      <td>4.0</td>
      <td>0.033</td>
      <td>0.847</td>
      <td>0.121</td>
      <td>0.9355</td>
      <td>May or May Not</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Top notch cafe inside of the cool Sci Arc camp...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Turkey Burger,Ghost Burger,Ghost Burger,Black ...</td>
      <td>5.0</td>
      <td>0.000</td>
      <td>0.760</td>
      <td>0.240</td>
      <td>0.9735</td>
      <td>Definitely</td>
    </tr>
    <tr>
      <th>24</th>
      <td>I had Semi booked because the brunch pictures ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Tomato Bisque,Tomato Bisque,Nitro Cold Brew,Ni...</td>
      <td>4.0</td>
      <td>0.025</td>
      <td>0.767</td>
      <td>0.208</td>
      <td>0.9972</td>
      <td>May or May Not</td>
    </tr>
  </tbody>
</table>
</div>




```python
Final_Reviews.review[15]
```




    "The Food  The food here is very good  The pastrami literally breaks into a billion pieces as soon as you put it in your mouth  The flavor is so subtle and juicy  it literally melts  I got the number 19  which is the pastrami sandwich with cole slaw and Swiss cheese on rye bread  I got two Latkas with cream and apple sauce  and the pastrami chili cheese fries  The sand which was awesome  the cole slaw was amazing  Creamy cold and crunchy  perfectly complimenting the flavors  The latkas were deep fried to perfection  And the chili cheese fries were huge  Like super huge  Had to take left over home The location  it's located in the center of west la  on Wiltshire  Traffic and parking can be a little tough but they have their own parking area wth validation  The restaurant is clean and nice The Service  as we walked in we were asked how big our party was  I was with my mom so just two  I was afraid it would take forever to get seated because the place was packed  But after a few minutes we were seated and promptly someone came over to take our drink order  The Service was nice  the lady was very attentive  refilled my drink and my mom coffee with out even asking  she just knew  and it was seamless  I didn't even notice it got refilled until she brought it food  Amazing  The food was starved relatively quickly and once you're done you go to the front and pay there "




```python

```
