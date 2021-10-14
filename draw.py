import numpy as np
from matplotlib import pyplot as plt

x = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]
y = [2, 3, 4 ,5, 6, 7, 8, 3, 4 ,5 ,6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 13, 8, 9, 10, 11, 12, 13, 14]
plt.scatter(x, y, linestyle="dashed")
plt.xlim(0, 10)
plt.ylim(0, 15)
ax = plt.gca()
ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)
plt.grid()
plt.show()