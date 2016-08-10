import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it

import scipy
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import pandas as pd
import math

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
#####################definitions###########################

def findPathsNoLC(G,u,n):
    if n==0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in findPathsNoLC(G,neighbor,n-1):
            if u not in path:
                paths.append([u]+path)
    return paths

def create_directed_graph(number_of_nodes, std_of_dist=1,center_of_dist = 0,list_of_edges = [],):

    num_nodes = number_of_nodes
    edge_list = list_of_edges
    cent_dist = center_of_dist
    std_dist = std_of_dist
    node_values_G = []
    G = nx.DiGraph()
    list_of_nodes = [n for n in range(num_nodes)]

    #for i in range(num_nodes):
       # node_values_G.append(np.random.normal(loc=0.0, scale=1.0))

    if(edge_list == []):
        edge_list_iterator = it.combinations(list_of_nodes, 2)
        for i in edge_list_iterator:
            edge_list.append(i)

    for ii in range(num_nodes):
        node_values_G.append(np.random.normal(cent_dist, std_dist))


    for i in range(len(list_of_nodes)):
        G.add_node(list_of_nodes[i], node_value=node_values_G[i])

    G.add_edges_from(edge_list)

    return G

def create_cloud_of_points(graph, dimension_of_cloud):
    cloud_dim = dimension_of_cloud
    G = graph
    list_of_paths = []
    list_of_points_in_n_space = []
    for g_node in G.nodes():
        list_of_paths.extend(findPathsNoLC(G, g_node,cloud_dim ))

    for n_walk in list_of_paths:
        temp_point_list = []
        for n_node in n_walk:
            temp_point_list.append(G.node[n_node]['node_value'])
        list_of_points_in_n_space.append(temp_point_list)

    df_points = pd.DataFrame( list_of_points_in_n_space)

    return df_points

def generate_save_point_clouds(cloud_name,number_of_clouds, number_of_nodes, dim_of_cloud, save_path = "C:\\Users\micha\PycharmProjects\\betti\cloud_data\\",stDev=1, center_of_dist = 0 ):

    cloudName = cloud_name
    numClouds = number_of_clouds
    numNodes = number_of_nodes
    dimOfCloud = (dim_of_cloud -1)
    savePath = save_path
    standardDev = stDev
    centerOfDist = center_of_dist

    for cloudNumber in range(numClouds):
        G_temp = create_directed_graph(numNodes,standardDev,centerOfDist)
        tempCloud = create_cloud_of_points(G_temp,dimOfCloud)
        tempCloud.to_csv(savePath + cloudName + str(cloudNumber) + ".csv", index=False)


##############################################

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
        class_indicator +=1

    df_total = pd.concat(list_of_betti_data)

    features = df_total.drop('TARGET',1)
    target = df_total['TARGET']

    return features, target







