import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#download DS from official car data website at this link https://archive.ics.uci.edu/dataset/19/car+evaluation  
#place data set in whichever file you choose then copy the file path and paste it between the quotation marks below

data = ''
df = pd.read_csv(data, header=None)

#print(df.shape)
#print(df.head())

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

#print(df.head())
print(df.info())


for col in col_names:
    print(df[col].value_counts())

#shows an output of all null places inside DS
df.isnull().sum()

x = df.drop(['class', 'doors'], axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(x_train.shape, x_test.shape)

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

rfc = RandomForestClassifier(n_estimators=100, random_state=0)

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

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
