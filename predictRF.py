from sklearn.ensemble import RandomForestClassifier
import numpy as np
import arcpy as arcpy
import arcpy.da as da
import pandas as pd
import matplotlib.pyplot as plt
import arcgisscripting as arc
import SSUtilities as utils
import os as os
import pydot
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

# make input data
input = r'preprop_data_layer'

# describe X as features
X = ['lineament_dens', 'drainage_dens', 'curah_hujan',
       'slope', 'elevasi',  'litologi_Alluvial',
       'litologi_Batigamping', 'litologi_Batuan_Sedimen',
       'litologi_Batugamping', 'litologi_Breksi_Gunung_Api',
       'litologi_Breksi_dan_Piroklastik_Muda',
       'litologi_Breksi_dan_Piroklastik_Tua', 'litologi_Volkanoklastik','potensi']
# describe y as target/class
y = ['potensi']

allvars = X + y

# convert data to shapefile
trainFC = da.FeatureClassToNumPyArray(input, ["SHAPE@XY"] + allvars)
spatRef = arcpy.Describe(input).spatialReference

data = pd.DataFrame(trainFC, columns = allvars)

# train test data split
X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.15, random_state=42)

# make model
clf = RandomForestClassifier(n_estimators=1000)

# train model
clf.fit(X_train, y_train)

# predict test from train data
y_pred = clf.predict(X_test)

# print accuracy
print(clf.score(X_test, y_test))
 
# print tree 
tree = clf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = X, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

# describe input for test data
input_test = r'test_data_Layer'

test_Data = da.FeatureClassToNumPyArray(input_test, ["SHAPE@XY"] + X)
spatRef_testData = arcpy.Describe(input).spatialReference

test_data_Train = pd.DataFrame(test_Data, columns= X)
potensi_pred = clf.predict(test_data_Train)

# export predict result to gdb (class 1)
nameFC = 'PotentionPrediction'
outputDir = r'D:\Document\ArcGIS\Projects\air\air.gdb'
potensi_Exist = test_Data[["SHAPE@XY"]][test_data_Train.index[np.where(potensi_pred==1)]]
arcpy.da.NumPyArrayToFeatureClass(potensi_Exist, os.path.join(outputDir, nameFC), ['SHAPE@XY'], spatRef_testData)

# export predict result to gdb (class 0)
nameFC0 = 'PotentionPrediction0'
potensi_doesntExist = test_Data[["SHAPE@XY"]][test_data_Train.index[np.where(potensi_pred==0)]]
arcpy.da.NumPyArrayToFeatureClass(potensi_doesntExist, os.path.join(outputDir, nameFC0), ['SHAPE@XY'], spatRef_testData)

