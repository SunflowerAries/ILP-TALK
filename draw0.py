import numpy as np
from matplotlib import pyplot as plt

x = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14]
y = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3 ,4, 0, 1, 2, 3 ,4, 5, 0, 1, 2, 3 ,4, 5, 6, 1, 2, 3 ,4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7]
plt.scatter(x, y, linestyle="dashed")
plt.xlim(0, 15)
plt.ylim(0, 10)
ax = plt.gca()
ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)
plt.grid()
plt.show()