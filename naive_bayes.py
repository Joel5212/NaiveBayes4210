#-------------------------------------------------------------------------
# AUTHOR: Joel Joshy
# FILENAME: native_bayes.py
# SPECIFICATION: Native Bayes Algorithm
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
import numpy
from sklearn.naive_bayes import GaussianNB
import csv


#reading the training data in a csv file
#--> add your Python code here
dbTraining = []
dbTest = []


with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTraining.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X = []
Y = []
confidenceCriteria = 0.75

for row in dbTraining:
        drow = []
        for i, value in enumerate(row):
            if i == 1: drow.append(1 if value == 'Sunny' else 2 if value == 'Overcast' else 3)
            elif i == 2: drow.append(1 if value == 'Hot' else 2 if value == 'Mild' else 3)
            elif i == 3: drow.append(1 if value == 'High' else 2)
            elif i == 4: drow.append(1 if value == 'Weak' else 2)
        X.append(drow)


#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
for row in dbTraining:
       Y.append(1 if row[-1] == 'Yes' else 2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)


#reading the test data in a csv file
#--> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
              if i > 0: #skipping the header
                dbTest.append(row)

#printing the header os the solution
#--> add your Python code here
print("Day".ljust(18) + "Outlook".ljust(18) + "Temperature".ljust(18) + "Humidity".ljust(18) + "Wind".ljust(18) + "PlayTennis".ljust(18) + "Confidence".ljust(18))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
featureNums = []
for row in dbTest:
        numrow = []
        for i, value in enumerate(row):
            # if i == 0: numrow.append(row[0])
            if i == 1: numrow.append(1 if value == 'Sunny' else 2 if value == 'Overcast' else 3)
            elif i == 2: numrow.append(1 if value == 'Hot' else 2 if value == 'Mild' else 3)
            elif i == 3: numrow.append(1 if value == 'High' else 2)
            elif i == 4: numrow.append(1 if value == 'Weak' else 2)
        featureNums.append(numrow)

print(" ")
for i in range(len(featureNums)):
    confidence = max(clf.predict_proba([featureNums[i]])[0])
    maxIndex = list(clf.predict_proba([featureNums[i]])[0]).index(confidence)

    if confidence >= confidenceCriteria:
        classResult = "Yes" if maxIndex == 0 else "No"
        print (str(dbTraining[i][0]).ljust(18) + str(dbTraining[i][1]).ljust(18) + str(dbTraining[i][2]).ljust(18) + str(dbTraining[i][3]).ljust(18) +
            str(dbTraining[i][4]).ljust(18) + str(classResult).ljust(18) + str(round(confidence,3)).ljust(18))



