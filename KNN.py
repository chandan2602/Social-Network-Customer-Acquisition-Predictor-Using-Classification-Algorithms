import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

df = pd.read_csv(r"D:\All Data-set\Social_Network_Ads.csv")

x = df.iloc[:,[2,3]].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

kn = KNeighborsClassifier(n_neighbors=4,p=1)
kn.fit(x_train,y_train)

y_pred = kn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True);


ac = accuracy_score(y_test, y_pred)
print('Accuracy Score::',ac*100)

bias = kn.score(x_train,y_train)
print('Bias::',bias*100)

variance = kn.score(x_test,y_test)
print('Variance::',variance*100) 

cr = classification_report(y_test, y_pred)
print(cr)