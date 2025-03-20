import pandas as pd
import numpy as np

def add_winner(row):
    if row['Home_Goals'] > row['Visitor_Goals']:
        return row['Home']
    elif row['Visitor_Goals'] > row['Home_Goals']:
        return row['Visitor']

def add_loser(row):
    if row['Home_Goals'] > row['Visitor_Goals']:
        return row['Visitor']
    elif row['Visitor_Goals'] > row['Home_Goals']:
        return row['Home']

def last_30_index(dataset, team, current_index):
    indices = []
    for i in range(current_index - 1, -1, -1):
        if dataset.loc[i, 'Home'] == team or dataset.loc[i, 'Visitor'] == team:
            indices.append(i)
        if len(indices) >= 30:
            break
    return indices

data25 = '/home/vm/PythonVenv/sci/Sci/Datasets/2024-25.csv'
data24 = '/home/vm/PythonVenv/sci/Sci/Datasets/2023-24.csv'
data23 = '/home/vm/PythonVenv/sci/Sci/Datasets/2022-23.csv'
data22 = '/home/vm/PythonVenv/sci/Sci/Datasets/2021-22.csv'
data21 = '/home/vm/PythonVenv/sci/Sci/Datasets/2020-21.csv'
data20 = '/home/vm/PythonVenv/sci/Sci/Datasets/2019-20.csv'
data19 = '/home/vm/PythonVenv/sci/Sci/Datasets/2018-19.csv'
data18 = '/home/vm/PythonVenv/sci/Sci/Datasets/2017-18.csv'
data17 = '/home/vm/PythonVenv/sci/Sci/Datasets/2016-17.csv'
data16 = '/home/vm/PythonVenv/sci/Sci/Datasets/2015-16.csv'


header = ['Date',  'Time', 'Visitor', 'Visitor_Goals', 'Home', 'Home_Goals', '','Attendence','LOG', 'Notes' ]

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
dataset25 = dataset25.drop(['Attendence','LOG','Notes'],axis=1)  

dataset25['Winner'] = dataset25.apply(add_winner, axis=1)
dataset25['Loser'] = dataset25.apply(add_loser, axis=1)
print(dataset25.head)

print(dataset25.iloc[last_30_index(dataset25, 'Toronto Maple Leafs', 600)])

