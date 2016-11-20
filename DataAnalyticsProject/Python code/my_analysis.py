import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Dataset from https://archive.ics.uci.edu/ml/datasets/Student+Performance

df = pd.read_csv('student_perf.csv')
print(df.head(3))

df.describe()

g1 = df['G1']
g2 = df['G2']
g3 = df['G3']
st = df['studytime']
tt = df['traveltime']
gt = df['goout']
health = df['health']
ab = df['absences']
Dalco = df['Dalc']
Walco = df['Walc']
roma = df['romantic']

plt.hist(g1, bins = 15)
plt.xlabel('G1 Marks')
plt.ylabel('No. of Students')
plt.show()

plt.hist(g2, bins = 15)
plt.xlabel('G2 Marks')
plt.ylabel('No. of Students')
plt.show()

plt.hist(g3, bins = 15)
plt.xlabel('G3 Marks')
plt.ylabel('No. of Students')
plt.show()

plt.plot(g3,st,'ro')
plt.axis(ymin = 0, ymax = 4.5, xmin = 0, xmax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('Amount of Study Time')
c = np.corrcoef(g3,st)
print("Correlation Matrix")
print(c)
plt.show()

plt.plot(g3,tt,'ro')
plt.axis(ymin = 0, ymax = 5, xmin = 0, xmax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('Amount of Travel Time')
c = np.corrcoef(g3,tt)
print("Correlation Matrix")
print(c)
plt.show()

plt.plot(g3,gt,'ro')
plt.axis(ymin = 0, ymax = 6, xmin = 0, xmax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('Amount of Go Out Time')
c = np.corrcoef(g3,gt)
print("Correlation Matrix")
print(c)
plt.show()

plt.plot(g3,health,'ro')
plt.axis(ymin = 0, ymax = 6, xmin = 0, xmax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('Health')
c = np.corrcoef(g3,health)
print("Correlation Matrix")
print(c)
plt.show()

plt.plot(g3,ab,'ro')
plt.axis(xmax = 22,ymax = 95)
plt.xlabel('G3 Marks')
plt.ylabel('No. of Absences')
c = np.corrcoef(g3,ab)
print("Correlation Matrix")
print(c)
plt.show()

#Useful for Performance Prediction
plt.plot(g3,g1,'ro')
plt.axis(xmax = 22,ymax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('G1 Marks')
c = np.corrcoef(g3,g1)
print("Correlation Matrix")
print(c)
plt.show()

#Useful for Performance Prediction
plt.plot(g3,g2,'ro')
plt.axis(xmax = 22,ymax = 22)
plt.xlabel('G3 Marks')
plt.ylabel('G2 Marks')
c = np.corrcoef(g3,g2)
print("Correlation Matrix")
print(c)
plt.show()

plt.hist(Dalco, bins = 15)
plt.xlabel('Workday Alcohol Consumption Level')
plt.ylabel('No. of Students')
plt.show()

#Strong Factor
plt.plot(g3,Dalco,'ro')
plt.axis(xmax = 22,ymax = 6, ymin = 0)
plt.xlabel('G3 Marks')
plt.ylabel('Workday Alcohol Consumption Level')
c = np.corrcoef(g3,Dalco)
print("Correlation Matrix")
print(c)
plt.show()

plt.hist(Walco, bins = 15)
plt.xlabel('Weekend Alcohol Consumption Level')
plt.ylabel('No. of Students')
plt.show()

#Weak Factor
plt.plot(g3,Walco,'ro')
plt.axis(xmax = 22,ymax = 6, ymin = 0)
plt.xlabel('G3 Marks')
plt.ylabel('Workday Alcohol Consumption Level')
c = np.corrcoef(g3,Walco)
print("Correlation Matrix")
print(c)
plt.show()

# 16-20: 1, 14-15: 2, 12-13: 3, 10-11: 4, 0-9: 5
A = [16,17,18,19,20]
B = [14,15]
C = [12,13]
D = [10,11]
F = [0,1,2,3,4,5,6,7,8,9]
G1_grades = []
G2_grades = []
G3_grades = []
for index, row in df.iterrows():
    score = row['G1']
    if score in A:
        G1_grades.append(1)
    elif score in B:
        G1_grades.append(2)
    elif score in C:
        G1_grades.append(3)
    elif score in D:
        G1_grades.append(4)
    else:
        G1_grades.append(5)

for index, row in df.iterrows():
    score = row['G2']
    if score in A:
        G2_grades.append(1)
    elif score in B:
        G2_grades.append(2)
    elif score in C:
        G2_grades.append(3)
    elif score in D:
        G2_grades.append(4)
    else:
        G2_grades.append(5)

for index, row in df.iterrows():
    score = row['G3']
    if score in A:
        G3_grades.append(1)
    elif score in B:
        G3_grades.append(2)
    elif score in C:
        G3_grades.append(3)
    elif score in D:
        G3_grades.append(4)
    else:
        G3_grades.append(5)

for index, row in df.iterrows():
    inter = df['internet']
    if inter is 'yes':
        df['inter'] = 1
    else:
        df['inter'] = 0

df['G1_grades'] = G1_grades
df['G2_grades'] = G2_grades
df['G3_grades'] = G3_grades

#DT All factors
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Medu'][i],df['Fedu'][i],df['Dalc'][i],df['Walc'][i],df['absences'][i],df['health'][i],df['traveltime'][i],df['inter'][i]])
g3 = list(df['G3_grades'])
X = g1g2
y = g3

clf = tree.DecisionTreeClassifier()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Mother education, Daily Alcohol, Travel Time
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Medu'][i],df['Dalc'][i],df['traveltime'][i],])
X = g1g2
y = g3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
clf = tree.DecisionTreeClassifier()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Daily Alcohol, Travel Time
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Dalc'][i],df['traveltime'][i]])
X = g1g2
y = g3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
clf = tree.DecisionTreeClassifier()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Travel Time, internet
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['traveltime'][i],df['inter'][i]])
X = g1g2
y = g3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
clf = tree.DecisionTreeClassifier()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#SVM
#All factors
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Medu'][i],df['Fedu'][i],df['Dalc'][i],df['Walc'][i],df['absences'][i],df['health'][i],df['traveltime'][i],df['inter'][i]])
g3 = list(df['G3_grades'])
X = g1g2
y = g3

clf = SVC()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Mother education, Daily Alcohol, Travel Time
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Medu'][i],df['Dalc'][i],df['traveltime'][i]])
X = g1g2
y = g3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
clf = SVC()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Daily Alcohol, Travel Time
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['Dalc'][i],df['traveltime'][i]])
X = g1g2
y = g3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
clf = SVC()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())

#Travel Time
g1g2 = []
for i in range(0,len(G1_grades)):
    g1g2.append([df['G1'][i],df['G2'][i],df['traveltime'][i],df['inter'][i]])
X = g1g2
y = g3
clf = SVC()
score = cross_val_score(clf,g1g2,g3)
print(score.mean())
