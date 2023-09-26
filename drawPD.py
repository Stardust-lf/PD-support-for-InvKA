import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from adjustText import adjust_text

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pickle
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r
plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=1200)
# fig,ax = plt.subplots(1,3,dpi=600)
models = ['Gait-RNNPart','MT3DCNN','GLN','GaitPart','GaitGL','3dLocalMT'
          ,'CSTL','RPNet','GaitSet','GaitNet','DyGait','GLN-HBS','GaitSet-HBS','GR-KD(Ours)']
alias = ['GR','MT','GL','GP','GG','LMT','CS','RP','GS','GN','DG','GH','GSH','GR-KD(Ours)']
dic_full = {'MT3DCNN':[90.1,83.9,90.9,96.4,95.7,98.5,97.5,97.5,99],'GaitGL':[92.7,87.8,93.9,97.6,97.3,98.8,98.3,99,97.9],
            'GaitSet':[90.4,86.5,83.3,95.2,98,94.5,99.4,97.9,96.9],'GR-KD(Ours)':[94.1,93.2,93.5,97.3,95,93.1,96,94.3,89.6]}
dic_half = {'Gait-RNNPart':[99.4,98.2,98],'GLN':[99.3,99.5,98.7],
            'GaitPart':[99.3,98.6,98.5],'3dLocalMT':[99,99.5,98.9],
            'CSTL':[99.4,99.2,98.4],'RPNet':[99,99.1,98.3],
            'GaitNet':[95.1,94.2,93.1],'DyGait':[99.2,98.9,98.3],
            'GLN-HBS':[99.4,98.9,98.9],'GaitSet-HBS':[99.2,97.4,98.3]}
FLOPS = [1.7,4.3,56,8.93,18.2,2.2,4.18,17,8.7,1.4,48,56,8.7,0.017]
method = [0,1,0,2,2,1,1,2,0,0,1,1,1,3]
dic_flop = {}
dic_method = {}
for i in range(len(models)):
    dic_flop[models[i]] = FLOPS[i]
    dic_method[models[i]] = method[i]
FLOPS = np.array(FLOPS,dtype=np.float)
print(len(FLOPS))
FLOPS*=1000000000
FLOPS = 10**(10-np.log10(FLOPS))
# ACC = [99.4,99.0,99.5,99.5,97.2,98.6,,96.7,95.0,92.3,98.4,96.5,96.0,96.0]
ACC = [99.4,99.0,99.5,99.3,99.0,99.5,99.4,99.0,99.4,94.2,99.2,99.4,99.2,98.0]
FLOPS_CON = [13,1.7,4.3,56,8.93,18.2,2.2,4.18,1.7,17,8.7,1.4,48,56,8.7,0.017]
FLOPS_CON = np.array(FLOPS_CON,dtype=np.float)

FLOPS_CON*=1000000000

# ax.set_yscale('log')
ax.set_ylabel('10^(10-log10(FLOPs))')
ax.set_xlabel('Accuracy on NM#1-4(%)')
#ax.set_ylim(-1,3)
ax.grid(True)
ax.set_xlim(80,100)

#fig,ax = plt.subplots()
mColor = ['dodgerblue','lime','gold','maroon']
mMarker = ['.','^','s','*']
for i in range(len(method)):
    method[i] = mColor[method[i]]
print(method)
#plt.errorbar(ACC[:-1],FLOPS[:-1],xerr=0,yerr=err.T,fmt='^',ecolor='gray',ms=0,color='white',elinewidth=1,capsize=4,alpha=0.7)
plt.scatter(x=ACC,y=FLOPS,color=method,marker='^',alpha=1,s=30)
# new_texts = [plt.text(x_, y_, text, fontsize=11) for x_, y_, text in zip(ACC, FLOPS, alias)]
# adjust_text(new_texts,ax=ax,
#             #only_move={'text': 'x'},
#             arrowprops=dict(arrowstyle='-', color='black',alpha=0),
#             )
plt.scatter(x=98.0,y=FLOPS[-1],marker='*',c='red',alpha=1,s=100)

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
axins = ax.inset_axes((0.15, 0.08, 0.5, 0.6))
axins.set_xlim([98.9,99.6])
axins.set_ylim([-0.8,7])
# axins.set_yscale("log")
axins.grid(True)
axins.scatter(x=ACC,y=FLOPS,color=method,marker='^',alpha=1,s=30)
print(10-np.log10(FLOPS))
# new_texts = [plt.text(x_, y_, text, fontsize=11) for x_, y_, text in zip(ACC, FLOPS, models)]
# adjust_text(new_texts,ax=axins,
#             only_move={'text': 'x'},
#             arrowprops=dict(arrowstyle='-', color='black',alpha=0),
#             )
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=0.5)
# for i in range(len(ACC)):
#     axins.text(ACC[i],FLOPS[i],alias[i])
#plt.show()
plt.savefig('SMLT_FLOP_ACC')
# plt.figure(dpi=1200)
#
# sns.heatmap(coef,cmap='Oranges')
# plt.xticks([])
# plt.yticks([])
# plt.savefig('VAR_NM')