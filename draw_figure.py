import matplotlib.pyplot as plt
plt.figure(figsize=(16, 4))
plt.subplot(141)
ax = plt.gca()
x = [10, 30, 50, 70]
y1 = [66.19, 74.93, 75.61,71.54]
y2 = [62.13, 76.19, 78.17, 74.59]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()

plt.plot([10, 30], [76.89, 79.97], '.r')
plt.plot(x, y1, 'r', label='$Weigths \sim N_k$')
plt.plot(x, y2, 'g', label='$Weigths \sim U_k$')

plt.hlines(79.16, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(78.13, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.ylabel("Accuaray")
plt.xlabel("% of Weights")
plt.title("Conv2")

plt.subplot(142)
ax = plt.gca()
y1 = [73.78, 83.3, 83.6, 78.98]
y2 = [72.45, 84.44, 86.07, 83.2]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()

plt.plot([10, 30], [82.28, 85.31], '.r')
plt.plot([10], [82.18], '.g')
plt.plot(x, y1, 'r', label='$Weigths \sim N_k$')
plt.plot(x, y2, 'g', label='$Weigths \sim U_k$')

plt.hlines(86.66, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(86.21, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv4")

plt.subplot(143)
ax = plt.gca()
y1 = [78.27, 86.05, 86.78, 83.41]
y2 = [73.99, 86.51, 88.17, 86.57]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()

plt.plot([10, 30], [83, 87.18], '.r')
plt.plot([10], [84.35], '.g')
plt.plot(x,y1,'r',label='$Weigths \sim N_k$')
plt.plot(x,y2,'g',label='$Weigths \sim U_k$')

plt.hlines(87.82, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(88.43, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv6")

plt.subplot(144)
ax = plt.gca()
y1 = [67.01, 87.11, 87.96, 84.63]
y2 = [0, 87.87, 89.2, 87.57]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()

plt.plot([10, 30], [84.67, 88], '.r')
plt.plot(x, y1, 'r', label='$Weigths \sim N_k$')
plt.plot([30, 50, 70], [87.87, 89.2, 87.57], 'g', label='$Weigths \sim U_k$')

plt.hlines(88.81, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(89.05, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv8")
plt.legend( bbox_to_anchor=(1,1), loc='center left')
# plt.legend(bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.show()