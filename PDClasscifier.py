import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV,StratifiedKFold,train_test_split
import sklearn.metrics as metrics
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

data = load_variavle('matrix64_PD_90')
sample = load_variavle('matrix64_nm_90')
y1 = [1]*len(data)
y2 = [0]*len(sample)
print(len(y1))
print(len(y2))
clf = LogisticRegression(max_iter=2000,C=100)
y_true = np.array(y1+y2,dtype=np.int)
full_data = np.concatenate([data,sample],axis=0)
full_data = np.reshape(full_data,[len(full_data),-1])
kf = KFold(n_splits=5, shuffle=True)
acc = []
for train_index, test_index in kf.split(full_data, y_true):
    x_train = full_data[train_index]
    y_train = y_true[train_index]
    x_test = full_data[test_index]
    y_test = y_true[test_index]
    print(len(x_test))
    print(len(x_test))
    clf.fit(x_train, y_train)
    acc.append(metrics.accuracy_score(y_test, clf.predict(x_test)))
print('avg acc is', np.average(acc))
