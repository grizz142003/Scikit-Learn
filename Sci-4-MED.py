import pandas as pd 
import numpy as np


class DecisionTree:
    
    def __init__(self, max_depth = 4, depth = 1):
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None
    
    def fit(self, data, target):
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)

    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for row in data.values])
    
    def __flow_data_thru_tree(self, row): 
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()
    
    def __calculate_impurity_score(self, data):
        if data is None or data.empty: return 0
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2
    
    def __find_best_split_for_column(self, col):
        x = self.data[col]
        unique_values = x.unique()
        if len(unique_values) == 1: return None, None
        information_gain = None
        split = None
        for val in unique_values:
            left = x <= val
            right = x > val
            left_data = self.data[left]
            right_data = self.data[right]
            left_impurity = self.__calculate_impurity_score(left_data[self.target])
            right_impurity = self.__calculate_impurity_score(right_data[self.target])
            score = self.__calculate_information_gain(left_count = len(left_data),
                                                      left_impurity = left_impurity,
                                                      right_count = len(right_data),
                                                      right_impurity = right_impurity)
            if information_gain is None or score > information_gain:
                information_gain = score
                split = val

            return information_gain, split
    def __calculate_information_gain(self, left_count, left_impurity, right_count, right_impurity):
        return self.impurity_score - ((left_count/len(self.data)) * left_impurity + (right_count/len(self.data)) * right_impurity)
    
    def __find_best_split(self):
        best_split = {}
        for col in self.independent:
            information_gain, split = self.__find_best_split_for_column(col)
            if split is None: continue
            if not best_split or best_split["information_gain"] < information_gain:
                best_split = {"split": split,
                              "col": col,
                              "information_gain": information_gain,}
        return best_split["split"], best_split["col"]
    
    def __create_branches(self):
        self.left = DecisionTree(max_depth = self.max_depth,
                                 depth = self.depth + 1)
        self.right = DecisionTree(max_depth = self.max_depth,
                                  depth = self.depth + 1)
        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.data[self.data[self.split_feature] > self.criteria]
        self.left.fit(data = left_rows, target = self.target)
        self.right.fit(data = right_rows, target = self.target)
        
    @property
    def is_leaf_node(self): return self.left is None

    @property
    def probability(self):
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for _, row in data.iterrows()])
    
    def __flow_data_thru_tree(self,row):
        if self.is_leaf_node: return self.probability
        tree = self.left if row[self.split_feature] <= self.criteria else self.right
        return tree.__flow_data_thru_tree(row)


train = pd.read_csv("./train_preprocessed.csv")
test = pd.read_csv("./test_preprocessed.csv")

model = DecisionTree()
model.fit(data = train, target = "Survived")
predictions = model.predict(test)

print(predictions[:5])