import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from catchCycle import cut,catchCycleForArray
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r



if __name__ == '__main__':
  for root, dir, files in os.walk(r'D:\PycharmProjects\proj1\gaitVer2\PD\ArrayPD'):
    for name in files:
      path = os.path.join(root,name)
      data = load_variavle(path)
      leng = int(len(data)/2)
      avg = np.average(np.average(data,axis=1),axis=1)
      start = np.where(avg>0)[0][0]
      end = np.where(avg>0)[0][-1]
      data = data[start+20:end-20]
      with open(path+"_cutted",'wb+') as f:
        pickle.dump(data,f)
        print(path)

