import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib import collections  as mc


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

N = 5

#menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = .3       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

standard = [147584,295168,295168,590080,26240]
rects1 = ax.bar(ind-width, standard, width, color='blue')#, yerr=menStd)

tensorized =   [25966,29566,29566,44896,3370]
rects2 = ax.bar(ind, tensorized, width, color='green')#, yerr=womenStd)

bayesiantensorized =   [8883,9651,8679,2600,2586]
rects3 = ax.bar(ind+width, bayesiantensorized, width, color='red')#, yerr=womenStd)
ax.set_yscale('log')
# add some
ax.set_ylabel('Parameters')
ax.set_xlabel('Layer')

ax.set_xticks(ind)
ax.set_xticklabels( ('2','3','4','5','6'))#, 'G3', 'G4', 'G5') )
plt.rcParams.update({'font.size': 21,'font.family':'Times New Roman'})
sns.despine()
ax.legend( (rects1[0], rects2[0],rects3[0]), ('BNN', 'BTNN','LR-TBNN') ,bbox_to_anchor=(.9,1.2),ncol = 3,prop={'size': 12})

ratios = np.divide(standard,bayesiantensorized)
lines = []
adj = [0,0,-0,30000,100]
for i in range(len(bayesiantensorized)):
    rect = rects3.get_children()[i]
    temp_rect = rects1.get_children()[i]
    mid_height = adj[i]+np.exp(np.log(rect.get_height())+(np.log(temp_rect.get_height())-np.log(rect.get_height()))/2)
    
    start = rect.get_x()
    lo = rect.get_height()
    hi = temp_rect.get_height()

    if i==0:
        lines = lines+[[(start, lo), (start+width, lo)], [(start, hi), (start+width, hi)],[(start+width/2, lo), (start+width/2, lo+mid_height-mid_height/3)],[(start+width/2, lo+1.5*mid_height), (start+width/2, hi)]]
   
    elif i ==1:
        lines = lines+[[(start, lo), (start+width, lo)], [(start, hi), (start+width, hi)],[(start+width/2, lo), (start+width/2, lo+mid_height-mid_height/5)],[(start+width/2, lo+1.5*mid_height), (start+width/2, hi)]]

    elif i ==2:
        lines = lines+[[(start, lo), (start+width, lo)], [(start, hi), (start+width, hi)],[(start+width/2, lo), (start+width/2, lo+mid_height-mid_height/5)],[(start+width/2, lo+mid_height+mid_height/2), (start+width/2, hi)]]
    elif i ==3:
        lines = lines+[[(start, lo), (start+width, lo)], [(start, hi), (start+width, hi)],[(start+width/2, lo), (start+width/2, lo+mid_height-mid_height/10)],[(start+width/2, lo+1.7*mid_height), (start+width/2, hi)]]
    elif i ==4:
        lines = lines+[[(start, lo), (start+width, lo)], [(start, hi), (start+width, hi)],[(start+width/2, lo), (start+width/2, lo+mid_height-mid_height/3)],[(start+width/2, lo+1.5*mid_height), (start+width/2, hi)]]        
     
    plt.text(rect.get_x() + rect.get_width()/2.0, mid_height, str((int)(ratios[i]))+'x', ha='center', va='bottom',fontsize = 20,weight = 'bold')

lc = mc.LineCollection(lines, colors='k', linewidths=2)
ax.add_collection(lc)

plt.show()
plt.figure()