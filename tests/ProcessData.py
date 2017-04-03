import matplotlib.pyplot as plt
import numpy as np

import dill


plt.ion()

meanData = []
minData = []
maxData = []
sampleSize = []

"""with open('meanData.dat', mode='rb') as f:
	meanData = dill.load(f)

with open('minData.dat', mode='rb') as f:
	minData = dill.load(f)

with open('maxData.dat', mode='rb') as f:
	maxData = dill.load(f)
"""
with open('approxDataTest.dat', mode='rb') as f:
	approxData = dill.load(f)

print(approxData)

with open('sampleSizeTest.dat', mode='rb') as f:
	sampleSize = dill.load(f)

"""fig = plt.figure(figsize=(14, 10), dpi=100)
ax = fig.add_subplot(1,1,1)
ax.violinplot(meanData, sampleSize, widths=2,
                      showmeans=True, showextrema=True, showmedians=True)
plt.show()

fig.savefig('montecarlo_mean.png', bbox_inches='tight', dpi=100)

fig = plt.figure(figsize=(14, 10), dpi=100)
ax = fig.add_subplot(1,1,1)
ax.violinplot(minData, sampleSize, widths=2,
                      showmeans=True, showextrema=True, showmedians=True)
plt.show()

fig.savefig('montecarlo_min.png', bbox_inches='tight', dpi=100)

fig = plt.figure(figsize=(14, 10), dpi=100)
ax = fig.add_subplot(1,1,1)
ax.violinplot(maxData, sampleSize, widths=2,
                      showmeans=True, showextrema=True, showmedians=True)
plt.show()

fig.savefig('montecarlo_max.png', bbox_inches='tight', dpi=100)


fig = plt.figure(figsize=(14, 10), dpi=100)
ax = fig.add_subplot(1,1,1)
ax.boxplot(meanData)
plt.show()

fig.savefig('montecarlo_mean_box.png', bbox_inches='tight', dpi=100)
"""
flierprops = dict(marker='o', markerfacecolor='#e7298a', alpha=0.5, markersize=12,
                  linestyle='none')


fig = plt.figure(figsize=(14, 10), dpi=100)
ax = fig.add_subplot(1,1,1)
bp = ax.boxplot(approxData, notch=True, flierprops=flierprops, 
  showmeans=True, meanline=True, patch_artist=True)

nSamples = np.asarray(sampleSize) / 5
ax.set_xticklabels(nSamples)

for box in bp['boxes']:
  # change outline color
  box.set( linewidth=2)
  # change fill color
  box.set( facecolor = 'lightblue' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
  whisker.set(linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
  cap.set( color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
  median.set( linewidth=2)

for m in bp['means']:
  m.set(linewidth=2)

plt.xlabel('Sample Density (Samples/1000 m^2)')
plt.ylabel('Average Relative Error (m/s)')
plt.title('Average Relative Error vs. Sample Density')

plt.show()



fig.savefig('montecarlo_approx_box.png', bbox_inches='tight', dpi=100)