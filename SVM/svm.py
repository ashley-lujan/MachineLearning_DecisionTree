import numpy as np

def stochastic_svm(training_data, tune_rate, C, epochs):
    d = train_data.shape[1]
    n = train_data.shape[0]
    w = np.zeros(d)
    for t in range(epochs): 
        train_x, train_y = shuffle(training_data)
        for xi, yi in zip(train_x, train_y):
            if yi * (w @ xi) <= 1: 
                #what the heck is [w0; 0]
                w = w - (yi * (tune_rate(t) * w)) + tune_rate(t) * C * n * yi * xi
            else: 
                w = (1 - tune_rate(t)) * w
    return w

def shuffle(train_data): 
    d = train_data.shape[1]
    np.random.shuffle(train_data)
    train_x = train_data[:, :(d-1)]
    train_y = train_data[:, (d - 1)]
    return train_x, train_y

if __name__ == "__main__":
    #use rate, use different c's
    train_data_filename = "datasets/bank-note/train.csv"
    test_data_filename = "datasets/bank-note/test.csv"

    test_data = np.genfromtxt(test_data_filename, delimiter=',', dtype=float) 
    train_data = np.genfromtxt(train_data_filename, delimiter=',', dtype=float) 

    d = train_data.shape[1]

    train_data[:, d - 1] = np.where(train_data[:, d - 1] == 0, -1, train_data[:, d-1])
    C = [100/873, 500/873, 700/873]
    def learning_rate_a(t, y0 = 1, a = 1):
        return (y0)/(1 + (y0/a) * t)
    stochastic_svm(train_data, learning_rate_a, C[0], 2)