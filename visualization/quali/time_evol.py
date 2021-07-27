import os, re, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams.update({'font.size': 20})

xo = np.array([32,64,128,256, 512, 1024])
ys = np.array([0.071,0.074,0.076,0.081,0.087, 0.092])
yv = np.array([0.079,0.088,0.102,0.112,0.115, 0.121])

fig, ax = plt.subplots(figsize=(10,4.5))


cs = ax.plot(xo, yv, lw=2, color='#9E0B0F', label=r'MMD$_{s}$')
#plt.axhline(y = 4.235, color = '#ffa700', linestyle = '-', label='Cross-View No truncation')
cv = ax.plot(xo, ys, lw=2, color='#2E477D', label=r'MMD$_{a}$')
#plt.axhline(y = 3.618, color = '#00ffff', linestyle = '-', label='Cross-Subject No truncation')

d = np.zeros(len(xo))
ax.fill_between(xo, yv, ys,  color='#CE0E14', alpha=0.8)
ax.fill_between(xo, ys, color='#426CA6', alpha=0.8)



plt.ylim(min(np.concatenate((ys,yv)))-0.02, max(np.concatenate((ys,yv)))+0.02)
plt.xlim(min(xo), max(xo))
plt.xticks(xo)
plt.xlabel('Temporal Length (Frames)')
plt.ylabel('MMD')

# Create offset transform by 5 points in x direction
dx = 5/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels()[:1]:
    label.set_transform(label.get_transform() - offset)


# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels()[1:3]:
    label.set_transform(label.get_transform() + offset)

plt.legend(prop={'size': 21})
plt.grid(True, color='#000', alpha=0.3, linewidth=2)
plt.tight_layout()
plt.savefig('time_evol.pdf')
plt.show()