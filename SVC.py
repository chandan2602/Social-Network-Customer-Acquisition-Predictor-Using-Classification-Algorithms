import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

df = pd.read_csv(r"D:\All Data-set\Social_Network_Ads.csv")

x = df.iloc[:,[2,3]]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

sc = SVC()

sc.fit(x_train,y_train)

cm = confusion_matrix(y_test,sc.predict(x_test))
sns.heatmap(cm,annot=True);

ac = accuracy_score(y_test,sc.predict(x_test))
print('Accuracy Score::',ac*100)

bias = sc.score(x_train,y_train)
print('Bias::',bias*100)

variance = sc.score(x_test,y_test)
print('Variance::',variance*100) 

cr = classification_report(y_test,sc.predict(x_test))
print(cr)