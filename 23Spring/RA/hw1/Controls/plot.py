import matplotlib.pyplot as plt
import csv
import numpy as np
  
x_1, y_1= [], []
x_2, y_2= [], []
interval= 1
  
with open('/home/jy/Desktop/cmu/robot autonomy/hw1/Controls/force_vs_time_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x_1.append(round(float(row[0]), 3))
        y_1.append(float(row[1]))

with open('/home/jy/Desktop/cmu/robot autonomy/hw1/Controls/force_vs_time_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x_2.append(round(float(row[0]), 3))
        y_2.append(float(row[1]))
    
fig, ax = plt.subplots()
ax.plot(x_1[::interval], y_1[::interval], c = 'b', label ='Force Control')
ax.plot(x_2[::interval], y_2[::interval], c = 'r', label ='Impedance Control')
# ax.axis('equal')

# plt.xlim(0, 10)
# plt.ylim(0, 20)
plt.grid()
# plt.xticks(np.arange(min(x_1), max(x_1)+1, 1.0))
# ax.set_xticklabels(x_1[::1], rotation=45)
print(x_1)
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Force vs Time: Oscillating whiteboard')
plt.legend()
plt.show()
