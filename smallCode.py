# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statistics as s

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
group = pd.read_csv('gender_submission.csv')


trainDat = pd.DataFrame(train,columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
testDat = pd.DataFrame(test,columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

myList=[]
for ind in trainDat['Embarked']:
    if pd.isnull(ind)==True:
        myList.append(0)
    elif ind == 'C':
        myList.append(1)
    elif ind == 'Q':
        myList.append(2)
    elif ind == 'S':
        myList.append(3)
    else:
        myList.append(4)
trainDat['Embarked']=myList
        
myList2=[]
for ind in trainDat['Sex']:
    if ind == 'male':
        myList2.append(0)
    elif ind == 'female':
        myList2.append(1)
    elif pd.isnull(ind):
        myList2.append(0)
trainDat['Sex']=myList2

myList=[]
for ind in testDat['Embarked']:
    if pd.isnull(ind)==True:
        myList.append(0)
    elif ind == 'C':
        myList.append(1)
    elif ind == 'Q':
        myList.append(2)
    elif ind == 'S':
        myList.append(3)
    else:
        myList.append(4)
testDat['Embarked']=myList
        
myList2=[]
for ind in testDat['Sex']:
    if ind == 'male':
        myList2.append(0)
    elif ind == 'female':
        myList2.append(1)
testDat['Sex']=myList2
        
groupDat = pd.DataFrame(train, columns = ['Survived'])

ageVecTrain = pd.DataFrame(trainDat, columns = ['Age'])
ageVecTest = pd.DataFrame(testDat, columns = ['Age'])
vecFare = pd.DataFrame(testDat,columns = ['Fare'])

shortAVTr = ageVecTrain.dropna()
shortAVTs = ageVecTest.dropna()
shortFare = vecFare.dropna()

avgVecTrain = shortAVTr.mean().round()
avgVecTest = shortAVTs.mean().round()
avgFare = shortFare.mean()

# Fill null values with average age values
trainDat['Age']=trainDat['Age'].fillna(avgVecTrain.Age.round(),inplace=False)
testDat['Age']=testDat['Age'].fillna(avgVecTest.Age.round(),inplace=False)
testDat['Fare']=testDat['Fare'].fillna(avgFare.Fare.round(),inplace=False)

### NEARNEST NEIGHBOR CLASSIFICATION
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()

trainX = np.array(trainDat)
#trainY = np.array(groupDat).astype(int)
#trainY = trainY.transpose()
trainY = np.array(groupDat[:])
                  
clf.fit(trainX, trainY)
testD=np.array(testDat).astype(int)
result = clf.predict(testDat)

groupJ = np.array(group)
outGroupDat = np.array(group)

groupS=groupJ[:,1]
index=0
accDec=0
for i in result:
    if i == groupS[index]:
        accDec=accDec+1
    outGroupDat[index][1] = i
    index=index+1
print('Accuracy of KNN:')
print((accDec/len(groupS))*100)

df = pd.DataFrame(outGroupDat,columns=[0,1])
df.to_csv('output2.csv')