# Daniel Moore
# 12/6/19
# This class will be used to transform and classify the training data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
import json
import numpy as np
import matplotlib.pyplot as plt

# svm Will take in x which is the result of the vectorizer
# and y which will be numerical labels for the data
# 0 is dankChristianMemes and 1 is Izlam
def getTrainedSVM():
    text = []  # This list will hold the text we'll vectorize
    labels = []

    f = open("dcm_comments_training_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(0)

    f.close()

    f = open("Izlam_comments_training_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(1)

    f.close()

    text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier( penalty='l2', random_state=42))])

    parameters_svm = {'clf-svm__alpha': [0.0001], 'clf-svm__class_weight': ['balanced'], 'clf-svm__eta0': [.5],
                      'clf-svm__learning_rate': ['invscaling'], 'clf-svm__loss': ['modified_huber'],
                      'clf-svm__power_t': [0.05], 'tfidf__use_idf': [True],
                      'vect__ngram_range': [(1, 2)]}

    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(text, labels)
    print("SVC Best Score Training " + str(gs_clf_svm.best_score_))
    # print(gs_clf_svm.best_params_)

    return gs_clf_svm


def testOnDevelopment(text_clf_svm, classifierName):
    text = []  # This list will hold the text we'll vectorize
    labels = []

    f = open("dcm_comments_development_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(0)

    f.close()

    f = open("Izlam_comments_development_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(1)

    f.close()

    predicted_svm = text_clf_svm.predict(text)
    print(classifierName + " Development mean accuracy " + str(np.mean(predicted_svm == labels)))


def runOnTestData(classifier, classifierName):
    text = []  # This list will hold the text we'll vectorize
    labels = []

    f = open("dcm_comments_test_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(0)

    f.close()

    f = open("Izlam_comments_test_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(1)

    f.close()

    predicted_svm = classifier.predict(text)
    print(classifierName + " Test mean accuracy " + str(np.mean(predicted_svm == labels)))

    precision, recall, thresholds = precision_recall_curve(labels, predicted_svm)
    plt.plot(recall, precision)
    plt.title(classifierName + " Test Data Precision-Recall Curve")
    plt.savefig(classifierName + "_Precision-Recall_Curve.png")
    plt.cla()


def getTrainedRandomForest():
    text = []  # This list will hold the text we'll vectorize
    labels = []

    f = open("dcm_comments_training_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(0)

    f.close()

    f = open("Izlam_comments_training_data.txt", "r")
    data = json.load(f)

    for d in data:
        text.append(d['body'])
        labels.append(1)

    f.close()

    text_clf_rf =  Pipeline([('vect', CountVectorizer(stop_words='english', analyzer='word')), ('tfidf', TfidfTransformer()),
                                 ('clf-rf', RandomForestClassifier(n_estimators=150, bootstrap=False,
                                                                   class_weight='balanced_subsample',
                                                                   criterion='entropy', max_features='log2'))])

    parameters_svm = {}

    gs_clf_rf = GridSearchCV(text_clf_rf, parameters_svm, n_jobs=-1)
    gs_clf_rf = gs_clf_rf.fit(text, labels)
    print("Random Forest Best Score training " + str(gs_clf_rf.best_score_))
    # print(gs_clf_svm.best_params_)

    return gs_clf_rf

if __name__ == "__main__":
    svm = getTrainedSVM()
    testOnDevelopment(svm, "SVC")
    runOnTestData(svm, "SVC")
    rf = getTrainedRandomForest()
    testOnDevelopment(rf,"Random Forest")
    runOnTestData(rf,"Random Forest")
