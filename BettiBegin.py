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
import os
import glob

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

details_of_experiment_class_1 = [100,10,3,1,0] #number of clouds, number of nodes, dimension of clouds, standard deviation, mean
details_of_experiment_class_2 = [100,10,3,2,0]
results_save="C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\results\\results_08-07-2017__seed_0__nornal_dist.csv" #modify this to dave file with details of exppariment

details_of_exp = details_of_experiment_class_1
details_of_exp.append(details_of_experiment_class_2[3])
details_of_exp.append(details_of_experiment_class_2[4])

bm.generate_save_point_clouds("funCloud",100,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\",1,0)
bm.generate_save_point_clouds("funCloud",100,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class2\\",2,0)



# Define command and arguments
command = 'Rscript'
path2script = 'Rcode\\TDA_betti1_1_0.R'

#args = ["C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\", "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_test.csv"]

# Build subprocess command
#cmd = [command, path2script] + args

#x = subprocess.check_output(cmd, universal_newlines=True)

path_to_class_1_data = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\"
path_to_save_class_1_data =  "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\betti_data\\bett_data_1.csv"

path_to_class_2_data = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class2\\"
path_to_save_class_2_data =  "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\betti_data\\bett_data_2.csv"


args_list =[[path_to_class_1_data, path_to_save_class_1_data], [path_to_class_2_data, path_to_save_class_2_data]]
for args in args_list:
    cmd =  [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)


#features, target = bm.prepare_data_for_machine_learning(["C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_1.csv", "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\bett_data_2.csv"])
features, target = bm.prepare_data_for_machine_learning([path_to_save_class_1_data, path_to_save_class_2_data])

features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=.33, random_state=0)


from sklearn.linear_model import LogisticRegression
#instantiate the classifier
clf_lr = LogisticRegression(C=1)


#fit the classifier
bm.train_and_evaluate(clf_lr,features_train, features_test,target_train,target_test,details_of_exp,results_save)



####file clean up###
def file_clean_up():
    folder_list = ["Class1", "Class2", "betti_data"]
    for folder in folder_list:
        filelist = glob.glob("C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\"+folder+"\\*.csv")
        for f in filelist:
            os.remove(f)



#file_clean_up()




