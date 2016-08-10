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


bm.generate_save_point_clouds("funCloud",10,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\p1\\data\\Class1\\",1,0)
bm.generate_save_point_clouds("funCloud",10,4,3,"C:\\Users\\micha\\PycharmProjects\\betti\\p1\\data\\Class2\\",2,0)