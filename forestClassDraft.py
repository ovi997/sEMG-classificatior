class semgClassifier():

    def __init__(self) -> None:
        self.path = './dataset.json'
        f = open(self.path)
        self.data = json.load(f)
        self.features = np.array(data['features'])
        self.labels = np.array(data['labels'])
        label_encoder = preprocessing.LabelEncoder() 
        self.labels = label_encoder.fit_transform(self.labels) 

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        featuresForest = self.arrangeDataset()
        forests = self.initializeForests(8)
        trainFeaturesList, testFeaturesList, trainLabelsList, testLabelsList = self.splitDataset(featuresForest)
        self.trainModel(trainFeaturesList, testFeaturesList, forests)

    
    def arrangeDataset(self):
        featuresForest = []
        for idx, feature in enumerate(self.features):
            for idxx in range(0, 8):
                if idx == 0:
                    featuresForest.append(feature[0])
                else: 
                    featuresForest[idxx] = np.vstack((featuresForest[idxx], feature[idxx]))
        return np.array(featuresForest)
    
    def initializeForests(self, numForests):
        forests = []
        for idx in range(numForests):
            rf = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=3, min_samples_leaf=3)
            forests.append(rf)
        return forests
    
    def splitDataset(self, featureForest):
        trainFeaturesList = []
        testFeaturesList = []
        trainLabelsList = []
        testLabelsList = []

        for idx in range(featureForest.shape[0]):

            trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
                featureForest[idx], self.labels, test_size = 0.3, random_state = 42)
            
            trainFeaturesList.append(trainFeatures)
            testFeaturesList.append(testFeatures)
            trainLabelsList.append(trainLabels)
            testLabelsList.append(testLabels)

        return trainFeaturesList, testFeaturesList, trainLabelsList, testLabelsList

    def trainModel(self, trainFeaturesList, trainLabelsList, forests):
        for idx, _ in enumerate(forests):
            forests[idx].fit(trainFeaturesList[idx], trainLabelsList[idx])
        print('Model trained!')
    
    def testModel(self, testFeaturesList, testLabelsList, forests):
        predictions = []
        for idx, forest in enumerate(forests):
            predictions.append(forest.predict(testFeaturesList[idx]))
        final_predictions = []
        decision = sum(predictions)
        if decision >= 4:
            final_predictions.append(1)
        else:
            final_predictions.append(0)
        
x = esmgClassifier()
x()
