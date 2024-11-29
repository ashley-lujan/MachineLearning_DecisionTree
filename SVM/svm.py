import numpy as np
from dual_svm import perform_dual_svm
from helpers import *

def stochastic_svm(training_data, tune_rate, C, epochs):
    #let w be w with bias vector
    d = training_data.shape[1]
    n = training_data.shape[0]
    w = np.zeros(d - 1) 
    # w0 = np.zeros(d - 2)
    for t in range(epochs): 
        train_x, train_y = shuffle(training_data)
        for xi, yi in zip(train_x, train_y):
            if yi * (w @ xi) <= 1: 
                w0 = np.insert(w[1:], 0, 0)
                w = w - (yi * (tune_rate(t) * w0)) + tune_rate(t) * C * n * yi * xi
            else: 
                w = (1 - tune_rate(t)) * w
    return w

def shuffle(train_data): 
    np.random.shuffle(train_data)
    return seperate_data(train_data)

def learning_rate_a(t, y0 = 0.1, a = 1):
    return (y0)/(1 + (y0/a) * t)

def learning_rate_b(t, y0 = 0.1):
    return (y0)/(1 + t)





def perform_primal_svm(train_data_filename, test_data_filename): 

    train_data = np.genfromtxt(train_data_filename, delimiter=',', dtype=float) 
    test_data = np.genfromtxt(test_data_filename, delimiter=',', dtype=float) 
    
    process_labels(np, train_data)
    process_labels(np, test_data)
    
    train_data = attach_ones(np, train_data)
    test_data = attach_ones(np, test_data)

    C = [100/873, 500/873, 700/873]
    learning_rates = [learning_rate_a, learning_rate_b]
    for learn_rate in learning_rates: 
        print("question")
        for c in C: 
            print("C = {}".format(c))
            w = stochastic_svm(train_data, learn_rate, C[0], 100)
            report_results(w, train_data, "train")
            report_results(w, test_data, "test")

if __name__ == "__main__":
    #use rate, use different c's

    train_data_filename = "datasets/bank-note/train.csv"
    test_data_filename = "datasets/bank-note/test.csv"
    #perform_primal_svm(train_data_filename, test_data_filename)
    perform_dual_svm(train_data_filename, test_data_filename)
    



    