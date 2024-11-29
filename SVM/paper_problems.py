import numpy as np
def main():
    w = np.array([0.06, 0.014875, -0.0449, -0.006095])
    x = np.array([1, 1.5, 0.2, -2.4])
    w0 = np.insert(w[1:], 0, 0)
    # print(1 - (1) * w @ x)
    gamma = 0.0025
    # print((w0 - 1 - 3 * x))
    print(w -  gamma * (w0 - 1 - 3 * x))


main()