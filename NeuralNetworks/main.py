from neurons import *
import numpy as np
import random

def learning_rate(t, y0 = 1, d = 0.1):
    return (y0 / (1 + (y0/d) * t))

def random_weights(num_of_weights): 
    mean = 0.0 
    std_dev = 1.0  

    
    weights = [random.gauss(mean, std_dev) for _ in range(num_of_weights)]
    return weights

def zero_weights(num_of_weights): 
    # mean = 0.0 
    # std_dev = 1.0  
    
    weights = [0 for _ in range(num_of_weights)]
    return weights

def stochasitc_gradient(data, width, T, w_initalizer):
    d = data.shape[1] #get number of features in data
    network = Network(d - 1, layers = 3, hidden_width=width, activator=sigmoid, y_activator=sgn)
    num_of_weights = len(network.getW())
    w = w_initalizer(num_of_weights)
    network.updateW(w)

    epsilon = 1e-4
    past_gradient = None
    gradient = None

    for epoch in range(T): 
        #shuffle dataset
        np.random.shuffle(data)
        for index, train_i in enumerate(data):
            if past_gradient is not None and getConvergence(gradient, past_gradient):
                print("converged at T = {} data i = {}".format(epoch, index))
                return network
        
            xi, yi = train_i[: d - 1], train_i[d - 1]
            loss = np.asarray(network.getGradient(xi, yi))
            w = np.asarray(network.getW())
            updated_w = w - (learning_rate(epoch) * (loss))
            network.updateW(updated_w)
            past_gradient = gradient
            gradient = loss
            
    return network

def getConvergence(gradient, past_gradient):
    epsilon = 1e-4
    s1 = np.linalg.norm(gradient)
    s2 = np.linalg.norm(past_gradient)
    return abs(s1 - s2) < epsilon

def getAccurac(test_data, network): 
    d = test_data.shape[1]
    n = test_data.shape[0]
    accurate = 0
    for test in test_data:
        xi, yi = test[: (d-1) ], test[(d - 1)]
        y_pred = network.predict(xi)
        if y_pred <= 0 and yi == 0:
            accurate += 1
        elif y_pred > 0 and yi == 1:
            accurate += 1
    return accurate / n



if __name__ == '__main__':
    #paper_question() #uncomment for proof
    widths = [5, 10, 25, 50, 100]
    train_data = np.genfromtxt('datasets/hw5/bank-note/train.csv', delimiter=',', dtype=float)
    test_data = np.genfromtxt('datasets/hw5/bank-note/test.csv', delimiter=',', dtype=float)

    
    T = 1000
    for width in widths: 
        network = stochasitc_gradient(train_data, width, T, zero_weights)
        train_accur = getAccurac(train_data, network)
        accuracy = getAccurac(test_data, network)
        print('at width: {}, train accuracy: {}, test accuracy : {}'.format(width, train_accur, accuracy))
        
