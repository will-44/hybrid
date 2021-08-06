import csv


import csv
import numpy as np
import matplotlib.pyplot as plt

data = []
posX = []
posXd = []
posY = []
posYd = []

with open('zfa.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        print(row[0])
        data.append(float(row[0]))
        posX.append(float(row[1]))
        posXd.append(float(row[2]))
        posY.append(float(row[3]))
        posYd.append(float(row[4]))

l = len(data)
print(l)
print(data)
plt.plot(np.arange(len(data))*0.05, np.array(data))
plt.xlabel("Secondes")
plt.ylabel("Newtons")
plt.show()

plt.plot(np.arange(len(data[20:]))*0.05, np.array(posX[20:]))
plt.plot(np.arange(len(data[20:]))*0.05, np.array(posXd[20:]))
plt.xlabel("Secondes")
plt.ylabel("Newtons")
plt.show()

plt.plot(np.arange(len(data[20:]))*0.05, np.array(posY[20:]))
plt.plot(np.arange(len(data[20:]))*0.05, np.array(posYd[20:]))
plt.xlabel("Secondes")
plt.ylabel("Newtons")
plt.show()

print(np.mean(data[100:]))
print(np.std(data[100:]))