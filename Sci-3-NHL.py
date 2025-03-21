import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import warnings
import kagglehub

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data25 = '/home/vm/PythonVenv/sci/Sci/2024-25MOD.csv'
data24 = '/home/vm/PythonVenv/sci/Sci/2023-24MOD.csv'
data23 = '/home/vm/PythonVenv/sci/Sci/2022-23MOD.csv'
data22 = '/home/vm/PythonVenv/sci/Sci/2021-22MOD.csv'
data21 = '/home/vm/PythonVenv/sci/Sci/2020-21MOD.csv'
data20 = '/home/vm/PythonVenv/sci/Sci/2019-20MOD.csv'
data19 = '/home/vm/PythonVenv/sci/Sci/2018-19MOD.csv'
data18 = '/home/vm/PythonVenv/sci/Sci/2017-18MOD.csv'
data17 = '/home/vm/PythonVenv/sci/Sci/2016-17MOD.csv'
data16 = '/home/vm/PythonVenv/sci/Sci/2015-16MOD.csv'


header = ['Date',  'Time', 'Visitor', 'Visitor_Goals', 'Home', 'Home_Goals', 'OT', 'Winner', 'Loser', 'HPP', 'VPP']

dataset25 = pd.read_csv(data25, skiprows=0)
dataset24 = pd.read_csv(data24, skiprows=0)
dataset23 = pd.read_csv(data23, skiprows=0)
dataset22 = pd.read_csv(data22, skiprows=0)
dataset21 = pd.read_csv(data21, skiprows=0)
dataset20 = pd.read_csv(data20, skiprows=0)
dataset19 = pd.read_csv(data19, skiprows=0)
dataset18 = pd.read_csv(data18, skiprows=0)
dataset17 = pd.read_csv(data17, skiprows=0)
dataset16 = pd.read_csv(data16, skiprows=0)

dataset25.columns = header
dataset24.columns = header
dataset23.columns = header
dataset22.columns = header
dataset21.columns = header
dataset20.columns = header
dataset19.columns = header
dataset18.columns = header
dataset17.columns = header
dataset16.columns = header

dataset24 = dataset24.dropna()
dataset24 = dataset24.iloc[300:].reset_index(drop=True)

x = dataset24.drop(['Visitor_Goals',  'Home_Goals','Date',  'Time','Winner', 'Loser','OT','Home', 'Visitor'], axis=1)
y = dataset24['Winner']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#print(x_train.shape, x_test.shape)

encoder = ce.OrdinalEncoder(cols=['HPP', 'VPP'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

rfc = RandomForestClassifier(n_estimators=1000, random_state=0)

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

feature_scores = pd.Series(rfc.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(feature_scores)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Importance of Safety Features')
#plt.show()


"""
No makey sense
"""
