import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"D:\All Data-set\Social_Network_Ads.csv")

x = df.iloc[:,[2,3]].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

ac = accuracy_score(y_test, y_pred)
print('accuracy score::',ac*100)

bias = dt.score(x_train,y_train)
print('bias::',bias*100)

variance = dt.score(x_test,y_test)
print('variance::',variance*100)

cr = classification_report(y_test, y_pred)
print(cr)
