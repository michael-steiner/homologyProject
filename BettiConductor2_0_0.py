import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import pandas as pd
import math

from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d
from sklearn.cross_validation import train_test_split

import subprocess

import BettiMod as bm

###Begin Work

bm.generate_save_point_clouds("funCloud",10,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\",1,0)
bm.generate_save_point_clouds("funCloud",10,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class2\\",2,0)


#generate_save_point_clouds(cloud_name,number_of_clouds, number_of_nodes, dim_of_cloud, save_path = "C:\\Users\micha\PycharmProjects\\betti\cloud_data\\",stDev=1, center_of_dist = 0 ):

name_of_cloud = "ten_nodes_normal_dist"
number_of_clouds = 1000
number_of_nodes = 10
dimension_of_cloud = 3
save_path_class_1 = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\"
save_path_class_2 = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class2\\"
standard_deviation_class_1 = 1
standard_deviation_class_2 = 2
mean_class_1 = 0
mean_class_2 = 1



#calling R code
command = 'Rscript'
path2script = 'Rcode\\TDA_betti1_1_0.R'


path_to_class_1_data = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\"
path_to_save_class_1_data =  "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_1.csv"

path_to_class_2_data = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class2\\"
path_to_save_class_2_data =  "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_2.csv"

args_list =[[path_to_class_1_data, path_to_save_class_1_data], [path_to_class_2_data, path_to_save_class_2_data]]
for args in args_list:
    cmd =  [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)

#machine learning
features, target = bm.prepare_data_for_machine_learning(["C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_1.csv", "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_2.csv"])

features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=.33, random_state=0)


from sklearn.linear_model import LogisticRegression
#instantiate the classifier
clf_lr = LogisticRegression(C=1)


#fit the classifier
bm.train_and_evaluate(clf_lr,features_train, features_test,target_train,target_test)



