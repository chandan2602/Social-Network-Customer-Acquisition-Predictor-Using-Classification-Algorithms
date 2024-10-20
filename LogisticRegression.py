import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"D:\All Data-set\logit classification.csv")

x = df.iloc[:,[2,3]].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.20,random_state=612)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

LR = LogisticRegression()
LR.fit(x_train,y_train)

y_pred = LR.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True);

ac = accuracy_score(y_test, y_pred)
print('Accuracy Score::',ac*100)

cr = classification_report(y_test, y_pred)
print(cr)

bias = LR.score(x_train, y_train)
print('Bias::',bias*100)

variance = LR.score(x_test, y_test)
print('Variance::',variance*100) 

df_1 = pd.read_csv(r"D:\All Data-set\Future prediction1.csv")

d2= df_1.copy()

df_1 = df_1.iloc[:,[2,3]].values
sc_1 = StandardScaler()
m = sc_1.fit_transform(df_1)


y_pred1 = pd.DataFrame()


d2 ['y_pred1'] = LR.predict(m)

d2.to_csv('pred_model.csv')

# To get the path 
import os
os.getcwd()
print(LR.score(x_test,y_test))











