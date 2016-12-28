import pandas as pd
import numpy as np
from numpy import *
import operator
import collections
import itertools
mul=pd.read_csv('C:/Users/chaithu/Desktop/mulabel.csv')
col_val=list(mul.columns.values)
h=len(col_val)
k=int(input("enter the number of labels"))
g=[]
for i in range(h-k,h):
    g.append(mul[col_val[i]])
for i in range(h-k,h):
    del mul[col_val[i]]    
v= pd.DataFrame(g)
labels=v.transpose()
col_val_1=list(labels.columns.values)
import time
start = time.time()
for i in range(0,k):
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.cross_validation import train_test_split
    x_train,x_test, y_train, y_test = train_test_split(mul,labels[col_val_1[i]], random_state = 1)
    from sklearn import tree
    from sklearn.metrics import classification_report, confusion_matrix
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(x_train, y_train)
    Test = clf_tree.predict(x_test)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(x_train, y_train)
    y_pred_class = knn.predict(x_test)
    from sklearn.ensemble import RandomForestClassifier
    rand_class = RandomForestClassifier(n_estimators=100)
    rand_class.fit(x_train, y_train)
    y_pred_class_rand = rand_class.predict(x_test)
    from sklearn.ensemble import VotingClassifier
    eclf = VotingClassifier(estimators=[('tree', clf_tree),('knn', knn),('rf',rand_class) ], voting='hard')
    for clf, label in zip([eclf], ['Ensemble']):
        Train = clf.fit(x_train, y_train)
        Test = clf.predict(x_test)
        e=pd.DataFrame(Test)
        confusion = confusion_matrix(y_test, Test)
        print('%s'%(confusion))
        from sklearn.metrics import accuracy_score
        t=accuracy_score(y_test,e)
        print(t)
        v=[]
        for i in range(0,k):
            v.append(t)
        target_names = ['class 0', 'class 1']
        print(classification_report(y_test,e, target_names=target_names))
end = time.time()
final_accuracy=sum(v)/k
print("Execution time of the program : " + str(end - start)) 
print("the final accuracy for collaborative mining multiclassification algorithm")
print("*********************************")
print(final_accuracy*100)
print("*********************************")
