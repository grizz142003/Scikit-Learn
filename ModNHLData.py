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

def Ppercentage(Wins, Losses, OTlosses):
    Ptotal = (Wins * 2) + OTlosses
    if (Wins+Losses+OTlosses) == 0:
        return 0
    else:
        return Ptotal / ((Wins+Losses+OTlosses) * 2)

def last_30(dataset, current_index):
    HTeam = dataset.loc[current_index,'Home']
    VTeam = dataset.loc[current_index,'Visitor']
    
    HWins, HLosses, HOTLosses = 0, 0, 0
    VWins,VOTLosses, VLosses = 0, 0, 0
    Hindices, Vindices = 0, 0

    for i in range(current_index - 1, -1, -1):
        if dataset.loc[i, 'Home'] == HTeam or dataset.loc[i, 'Visitor'] == HTeam:
            if dataset.loc[i, 'Winner'] == HTeam:
                HWins += 1 
                Hindices += 1
            elif dataset.loc[i, 'Loser'] == HTeam and pd.isna(dataset.loc[i, 'OT']):
                HLosses += 1 
                Hindices += 1
            elif dataset.loc[i, 'Loser'] == HTeam and (dataset.loc[i,'OT'] == 'OT' or dataset.loc[i,'OT'] == 'SO'):
                HOTLosses += 1
                Hindices += 1
        if Hindices >= 30:
            break
    
    for i in range(current_index - 1, -1, -1):
        if dataset.loc[i, 'Home'] == VTeam or dataset.loc[i, 'Visitor'] == VTeam:
            if dataset.loc[i, 'Winner'] == VTeam:
                VWins += 1 
                Vindices += 1
            elif dataset.loc[i, 'Loser'] == VTeam and pd.isna(dataset.loc[i, 'OT']):
                VLosses += 1 
                Vindices += 1
            elif dataset.loc[i, 'Loser'] == VTeam and (dataset.loc[i,'OT'] == 'OT' or dataset.loc[i,'OT'] == 'SO'):
                VOTLosses += 1
                Vindices += 1
        if Vindices >= 30:
            break
    HPoint = Ppercentage(HWins, HLosses, HOTLosses)
    VPoint = Ppercentage(VWins, VLosses, VOTLosses)

    return HPoint, VPoint

def ModifyDS(dataset):
    dataset['Winner'] = dataset.apply(add_winner, axis=1)
    dataset['Loser'] = dataset.apply(add_loser, axis=1)
    dataset['HPP'] = 0.0
    dataset['VPP'] = 0.0
    
    for i in range(len(dataset)):
        dataset.at[i, 'HPP'], dataset.at[i, 'VPP'] = last_30(dataset, i)

    return dataset

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


header = ['Date',  'Time', 'Visitor', 'Visitor_Goals', 'Home', 'Home_Goals', 'OT','Attendence','LOG', 'Notes' ]

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


dataset25 = dataset25.drop(['Attendence','LOG','Notes'],axis=1)  
dataset24 = dataset24.drop(['Attendence','LOG','Notes'],axis=1)  
dataset23 = dataset23.drop(['Attendence','LOG','Notes'],axis=1)  
dataset22 = dataset22.drop(['Attendence','LOG','Notes'],axis=1)  
dataset21 = dataset21.drop(['Attendence','LOG','Notes'],axis=1)  
dataset20 = dataset20.drop(['Attendence','LOG','Notes'],axis=1)  
dataset19 = dataset19.drop(['Attendence','LOG','Notes'],axis=1)  
dataset18 = dataset18.drop(['Attendence','LOG','Notes'],axis=1)  
dataset17 = dataset17.drop(['Attendence','LOG','Notes'],axis=1)  
dataset16 = dataset16.drop(['Attendence','LOG','Notes'],axis=1)  



dataset25 = ModifyDS(dataset25)
dataset24 = ModifyDS(dataset24)
dataset23 = ModifyDS(dataset23)
dataset22 = ModifyDS(dataset22)
dataset21 = ModifyDS(dataset21)
dataset20 = ModifyDS(dataset20)
dataset19 = ModifyDS(dataset19)
dataset18 = ModifyDS(dataset18)
dataset17 = ModifyDS(dataset17)
dataset16 = ModifyDS(dataset16)


dataset25.to_csv("2024-25MOD.csv", index=False)
dataset24.to_csv("2023-24MOD.csv", index=False)
dataset23.to_csv("2022-23MOD.csv", index=False)
dataset22.to_csv("2021-22MOD.csv", index=False)
dataset21.to_csv("2020-21MOD.csv", index=False)
dataset20.to_csv("2019-20MOD.csv", index=False)
dataset19.to_csv("2018-19MOD.csv", index=False)
dataset18.to_csv("2017-18MOD.csv", index=False)
dataset17.to_csv("2016-17MOD.csv", index=False)
dataset16.to_csv("2015-16MOD.csv", index=False)


print(dataset25.head)


