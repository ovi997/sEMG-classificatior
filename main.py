import json
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import svm

f = open('./dataset.json')
data = json.load(f)

features = np.array(data['features'])
labels = np.array(data['labels'])

label_encoder = preprocessing.LabelEncoder() 
labels = label_encoder.fit_transform(labels) 
features_forest = []

for idx, feature in enumerate(features):
    for idxx in range(0, 8):
        if idx == 0:
            features_forest.append(feature[0])
        else: 
            features_forest[idxx] = np.vstack((features_forest[idxx], feature[idxx]))


train_features, test_features, train_labels, test_labels = train_test_split(features_forest[0], labels, test_size = 0.3, random_state = 42)
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(features_forest[1], labels, test_size = 0.3, random_state = 42)
train_features3, test_features3, train_labels3, test_labels3 = train_test_split(features_forest[2], labels, test_size = 0.3, random_state = 42)
train_features4, test_features4, train_labels4, test_labels4 = train_test_split(features_forest[3], labels, test_size = 0.3, random_state = 42)
train_features5, test_features5, train_labels5, test_labels5 = train_test_split(features_forest[4], labels, test_size = 0.3, random_state = 42)
train_features6, test_features6, train_labels6, test_labels6 = train_test_split(features_forest[5], labels, test_size = 0.3, random_state = 42)
train_features7, test_features7, train_labels7, test_labels7 = train_test_split(features_forest[6], labels, test_size = 0.3, random_state = 42)
train_features8, test_features8, train_labels8, test_labels8 = train_test_split(features_forest[7], labels, test_size = 0.3, random_state = 42)

rf1 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf1.fit(train_features, train_labels)

rf2 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf2.fit(train_features2, train_labels2)

rf3 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf3.fit(train_features3, train_labels3)

rf4 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf4.fit(train_features4, train_labels4)

rf5 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf5.fit(train_features5, train_labels5)

rf6 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf6.fit(train_features6, train_labels6)

rf7 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf7.fit(train_features7, train_labels7)

rf8 = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
rf8.fit(train_features8, train_labels8)

predictions = rf1.predict(test_features)
predictions2 = rf2.predict(test_features2)
predictions3 = rf3.predict(test_features3)
predictions4 = rf4.predict(test_features4)
predictions5 = rf5.predict(test_features5)
predictions6 = rf6.predict(test_features6)
predictions7 = rf7.predict(test_features7)
predictions8 = rf8.predict(test_features8)

final_predictions = []
for idx in range(0, len(predictions)):
    decision = predictions[idx] + predictions2[idx] + predictions3[idx] + predictions4[idx] +  predictions5[idx] + predictions6[idx] + predictions7[idx] + predictions8[idx]
    if decision >= 4:
        final_predictions.append(1)
    else:
        final_predictions.append(0)


print('Actual labels\n', test_labels.tolist())
print('Predicted labels\n', final_predictions, '\n')

errors = abs(final_predictions - test_labels)
print('Errors:', errors,'\n')

print('Accuracy:', round((len(errors) - sum(errors))/len(errors)*100, 2), '%.\n')

active_TP = 0
fatigue_TP = 0
active_FP = 0
fatigue_FP = 0
active_TN = 0
fatigue_TN = 0
active_FN = 0
fatigue_FN = 0 

for idx in range(len(final_predictions)):
    if final_predictions[idx] == 0 and test_labels[idx] == 0:
        active_TP += 1
        fatigue_TN += 1
    if final_predictions[idx] == 0 and test_labels[idx] == 1:
        active_FP += 1
        fatigue_FN += 1
    if final_predictions[idx] == 1 and test_labels[idx] == 0:
        active_FN += 1
        fatigue_FP += 1
    if final_predictions[idx] == 1 and test_labels[idx] == 1:
        active_TN += 1
        fatigue_TP += 1
    

print('Precision - active:', active_TP/ (active_TP + active_FP))
print('Recall - active:', active_TP/ (active_TP + active_FN), '\n')

print('Precision - fatigue:', fatigue_TP/ (fatigue_TP + fatigue_FP))
print('Recall - fatigue:', fatigue_TP/ (fatigue_TP + fatigue_FN))