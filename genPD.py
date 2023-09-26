import cv2
import numpy as np
import os
import pickle
from Utils import load_variavle
def catchST(filepath):
    result = []
    vc = cv2.VideoCapture(filepath)
    count = 0
    while vc.isOpened():
        # if count % 2 == 0:
        #     count += 1
        #     continue
        # get a frame
        ret, frame = vc.read()
        if frame is None:
            break
        # show a frame
        if ret == True:
            frame = cv2.resize(frame,(400,400))
            ret,frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
            result.append(np.array(frame))
            count+=1
    vc.release()
    result = np.array(result,dtype=np.float)
    return result

if __name__ == '__main__':
    for root, dirs, files in os.walk("./PD/videos", topdown=False):
        for name in files:
            print(name)
            data=catchST(os.path.join(root, name))
            with open('PDdata_'+name,'wb+') as f:
                pickle.dump(data,f)


