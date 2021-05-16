import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.arange(1, 2000)
    eps_start = 0.95
    eps_end = 0.02
    eps_decay = 400
    y = eps_end + (eps_start - eps_end) * np.exp(-1. * (x / eps_decay))

    plt.plot(x, y)
    plt.show()