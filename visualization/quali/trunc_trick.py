import os, re, pickle
import numpy as np
import matplotlib.pyplot as plt  

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams.update({'font.size': 18})

xo = np.array([1.00,0.95,0.90,0.85,0.8])
ys = np.array([3.529,3.473,3.538,3.841,4.286])
yv = np.array([4.1178,4.072,4.224,4.673,4.989])

fig, ax = plt.subplots(figsize=(10,4.5))


cs = ax.plot(xo, yv, lw=2, color='#9E0B0F', label='Cross-View')
plt.axhline(y = 4.235, lw=2, color = '#ff9a00', zorder=10, linestyle = '-', label=r'$\hookrightarrow$ No truncation')
cv = ax.plot(xo, ys, lw=2, color='#2E477D', label='Cross-Subject')
plt.axhline(y = 3.618, lw=2, color = '#00ff83', linestyle = '-', label=r'$\hookrightarrow$ No truncation')

d = np.zeros(len(xo))
ax.fill_between(xo, yv, ys, color='#CE0E14', alpha=0.8)
ax.fill_between(xo, ys, color='#426CA6', alpha=0.8)


ax.hlines(min(yv), xmin=0, xmax=xo[np.where(yv==min(yv))], lw=2, color='#CCD1D7', linestyle='dashed')
ax.hlines(min(ys), xmin=0, xmax=xo[np.where(ys==min(ys))], lw=2, color='#CCD1D7', linestyle='dashed')

ax.vlines(xo[np.where(yv==min(yv))], ymin=0, ymax=min(yv), lw=2, color='#CCD1D7', linestyle='dashed')
ax.vlines(xo[np.where(ys==min(ys))], ymin=0, ymax=min(ys), lw=2, color='#CCD1D7', linestyle='dashed')



plt.ylim(min(np.concatenate((ys,yv)))-0.5, max(np.concatenate((ys,yv)))+0.5)
plt.xlim(min(xo), max(xo))
plt.xticks(xo)
plt.xlabel('Truncation Threshold')
plt.ylabel('FID')

plt.legend(prop={'size': 18})
plt.grid(True, color='#000', alpha=0.3)
plt.tight_layout()
plt.savefig('trunc_evol.pdf')
plt.show()