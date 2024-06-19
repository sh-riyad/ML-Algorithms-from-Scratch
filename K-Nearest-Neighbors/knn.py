import pandas as pd
import numpy as np

class knn:
    def __init__(self,k=1):
        self.k = k
        
    def fit(self,X_train,y_train):
        self.X_train = np.array(X_train)
        self.y_train = pd.DataFrame(y_train) # converting in dataframe if inputs comes in array
        
    def predict(self,X_test):
        self.X_test = np.array(X_test)
        
        distance_matrix = np.zeros((self.X_train.shape[0],self.X_test.shape[0]))
        # calculating distance
        for i in range(self.X_train.shape[0]):
            for j in range(self.X_test.shape[0]):
                # distance = np.sqrt((self.X_test[j][0] - self.X_train[i][0])**2 +  (self.X_test[j][1] - self.X_train[i][1])**2)
                distance = np.sqrt(np.sum((self.X_test[j] - self.X_train[i])**2))
                distance_matrix[i][j] = round(distance,2)
                
        distance_matrix = pd.DataFrame(distance_matrix)
        distance_matrix = pd.concat([distance_matrix, self.y_train],axis=1)
        
        prediction = []
        for i in range(self.X_test.shape[0]):
            k_min_distances = np.sort(distance_matrix.iloc[:,i])[:self.k]
            prediction.append(distance_matrix[distance_matrix.iloc[:,i].isin(k_min_distances)].iloc[:,-1].mode()[0])
        
        return prediction