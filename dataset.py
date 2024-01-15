from typing import Any
from features import SignalProcessing
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

class computeDataset():
    def __init__(self, databasePath) -> None:
        self.databasePath = databasePath
        self.dataset = {'features': [], 'labels': []}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        list_of_files = []
        for (dirpath, dirnames, filenames) in os.walk(self.databasePath):
            for filename in filenames:
                if filename.split('_')[2] == '2':
                    list_of_files.append(filename)
        print('Start processing signals...')
        i=0
        for f in list_of_files:
            i+=1
            filePath = self.databasePath + '\\'+ f
            print(f'Signal {i}/{len(list_of_files)}')
            self.appendDataset(filePath)
        self.generateJSON()

    def appendDataset(self, filePath):
        extractFeatures = SignalProcessing(filePath)
        activeMuscleFeatures, fatigueMuscleFeatures = extractFeatures()
        self.dataset['features'].append(activeMuscleFeatures.tolist())
        self.dataset['labels'].append('active')
        self.dataset['features'].append(fatigueMuscleFeatures.tolist())
        self.dataset['labels'].append('fatigue')
        
    def generateJSON(self):
        with open('dataset.json', 'w') as f:
            json.dump(self.dataset, f)



class plotData():
    def __init__(self, filepath) -> None:
        self.filePATH = filepath
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        data = self.readJSON()
        for index, item in enumerate(data['features']):
            spec = np.array(item).astype(float)
            print(data['labels'][index])
            for idx in range(0, spec.shape[0]):
                copy = spec[idx].astype(float)
                ranged = max(copy) - min(copy)
                for idxx in range(0, spec[idx].shape[0]):
                    spec[idx][idxx] = (spec[idx][idxx].astype(float) - min(copy) )/ ranged
            print(index)
            plt.figure()
            plt.title(f"Feature Vector {data['labels'][index].capitalize()} Muscle")
            s = sns.heatmap(data=spec, vmin=0, vmax=1)
            s.set_xlabel('Features', fontsize=10)
            s.set_ylabel('Channel', fontsize=10)
            plt.show()


    def readJSON(self):
        f = open('dataset.json')
        data = json.load(f)

        return data

    

# x = computeDataset('F:\MASTER\proiect TB\database')
# x()

y = plotData('F:\MASTER\proiect TB\dataset.json')
y()