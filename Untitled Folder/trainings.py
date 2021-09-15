import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle



def data_split(data1 , ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data1))
    test_size = int(len(data1) * ratio)
    train_indecies = shuffled[:test_size]
    train_test = shuffled[test_size:]
    return data1.iloc[train_test] ,data1.iloc[train_indecies]
    


if __name__ == '__main__':
    
    #read data
    df = pd.read_csv('Data1.csv')
    train, test = data_split(df , 0.2)
    x_train =train[['Fever','Bodypain','Age','Runnynose','DifBreathe']].to_numpy()
    x_test =test[['Fever','Bodypain','Age','Runnynose','DifBreathe']].to_numpy()

    y_train =train[['InfectionProb']].to_numpy().reshape(1604,)
    y_test =test[['InfectionProb']].to_numpy().reshape(400,)
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(x_train,y_train)

    #to open file where to store data
    file = open('model.pkl','wb')

    #Dump information to a file
    pickle.dump(clf ,file)
    file.close()

    
  




