import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from catchCycle import cut,catchCycleForArray,sepCycleArray


def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

if __name__ == '__main__':
    # result = []
    # for root, dir, files in os.walk(r'D:\PycharmProjects\proj1\gaitVer2\PD\Cutted'):
    #     for name in files:
    #         path = os.path.join(root,name)
    #         data = load_variavle(path)
    #         result.append(data)
    #         # print(data.shape)
    #         # semilArr,cutPos = catchCycleForArray(data)
    #         # plt.plot(semilArr)
    #         # plt.show()
    #         #break
    # with open('CutPd','wb+') as f:
    #     pickle.dump(result,f)
    data = load_variavle('CutPd')
    # print(data[0].shape)
    # print(data[1].shape)
    # data[1] = data[1][65:165]
    # data[3] = data[3][25:95]
    # data[4] = data[4][25:]
    # data[6] = data[6][17:95]
    # data[9] = data[9][55:160]
    # data[11] = data[11][50:]
    # data[12] = data[12][:130]
    # data[14] = data[14][25:120]
    # data[15] = data[15][20:90]
    # data[16] = data[16][25:175]
    # data[17] = data[17][40:]
    # data[20] = data[20][:140]
    # data[21] = data[21][:65]
    # data[22] = data[22][55:200]
    # with open('CutPd','wb+') as f:
    #     pickle.dump(data,f)
    data[12] = data[12][20:]
    data[13] = data[13][20:]
    result = []
    for i in range(len(data)):
        print(i)
        imgSlipArr,centersArr = sepCycleArray(data[i])
        print(imgSlipArr.shape)
        result.append(imgSlipArr)
        print(len(result))
    with open('PDCuttedData','wb+') as f:
        pickle.dump(result,f)