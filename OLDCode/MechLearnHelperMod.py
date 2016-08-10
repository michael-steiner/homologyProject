import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d
from sklearn.cross_validation import train_test_split


########################Functions###################
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print("Accuaracy on training set:")
    print(clf.score(X_train, y_train))

    print("Accuaracy on test set:")
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test, y_pred))

    cm = (metrics.confusion_matrix(y_test, y_pred))

    cm_df = pd.DataFrame(cm, index=['Predicted Class 0', 'Predicted Class 1'],
                         columns=['Actual Class 0', 'Actual Class 1'])

    print(cm)
    return cm


def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])

    roc_auc = auc(fpr, tpr)

    p = figure(title='Receiver Operating Characteristic')
    # Plot ROC curve
    p.line(x=fpr, y=tpr, legend='ROC curve (area = %0.3f)' % roc_auc)
    p.x_range = Range1d(0, 1)
    p.y_range = Range1d(0, 1)
    p.xaxis.axis_label = 'False Positive Rate or (1 - Specifity)'
    p.yaxis.axis_label = 'True Positive Rate or (Sensitivity)'
    p.legend.orientation = "bottom_right"
    show(p)


# employ K-fold cross validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem


def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))


def prepare_data_for_machine_learning(betti_data_paths):
    list_of_betti_data = []
    class_indicator = 0

    for betti_data_path in betti_data_paths:

        temp_df = (pd.DataFrame.from_csv(betti_data_path))
        temp_df['TARGET'] = class_indicator

        list_of_betti_data.append(temp_df)

    df_total = pd.concat(list_of_betti_data,1)

    features = df_total.drop('TARGET',1)
    target = df_total['TARGET']

    return features, target



path_list = ["C:\\Users\\micha\\PycharmProjects\\betti\\betti_data\\bettiCurvesClass1New2.csv","C:\\Users\\micha\\PycharmProjects\\betti\\betti_data\\bettiCurvesClass2New2.csv" ]

f, t = prepare_data_for_machine_learning(path_list)

print(t)



