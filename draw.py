import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

x = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
y = [1, 2, 3 ,4, 2 ,3 ,4, 3, 4, 4]
plt.scatter(x, y, linestyle="dashed")
plt.xlabel('i')
plt.ylabel('j')
plt.xlim(0, 5)
plt.ylim(0, 5)
ax = plt.gca()
ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)
plt.grid()
plt.show()