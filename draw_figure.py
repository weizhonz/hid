import matplotlib.pyplot as plt
plt.figure(figsize=(16, 4))
plt.subplot(141)
ax = plt.gca()
x = [30, 50, 70]
y1 = [74.66, 75.61,71.54]
y2 = [76.19, 78.17, 74.59]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()
plt.plot(x,y1,'r',label='$Weigths \sim N_k$')
plt.plot(x,y2,'g',label='$Weigths \sim U_k$')

plt.hlines(79.16, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(78.13, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.ylabel("Accuaray")
plt.xlabel("% of Weights")
plt.title("Conv2")

plt.subplot(142)
ax = plt.gca()
y1 = [83.59, 83.6, 78.98]
y2 = [84.66, 86.07, 83.2]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()
plt.plot(x,y1,'r',label='$Weigths \sim N_k$')
plt.plot(x,y2,'g',label='$Weigths \sim U_k$')

plt.hlines(86.66, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(86.21, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv4")

plt.subplot(143)
ax = plt.gca()
y1 = [85.81, 86.78, 83.41]
y2 = [86.51, 88.17, 86.57]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()
plt.plot(x,y1,'r',label='$Weigths \sim N_k$')
plt.plot(x,y2,'g',label='$Weigths \sim U_k$')

plt.hlines(87.82, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(88.43, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv6")

plt.subplot(144)
ax = plt.gca()
y1 = [87.09, 87.96, 84.63]
y2 = [87.87, 89.2, 87.57]
plt.xlim(0, 100)
plt.ylim(65, 90)
ax.invert_xaxis()
plt.plot(x,y1,'r',label='$Weigths \sim N_k$')
plt.plot(x,y2,'g',label='$Weigths \sim U_k$')

plt.hlines(88.81, 0, 100, colors = "b", label='Learned Dense Weights (Adam)', linestyles = "dashed")
plt.hlines(89.05, 0, 100, colors = "black", label='Learned Dense Weights (SGD)', linestyles = "dashed")
plt.xlabel("% of Weights")
plt.title("Conv8")
plt.legend( bbox_to_anchor=(1,1), loc='center left')
# plt.legend(bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.show()