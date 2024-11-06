import numpy as np

def saveDataSet(filename):
    result = []
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            result.append(terms)
    return result

def perception_update(train_x, train_y, w, r): 
    for xi, yi in zip(train_x, train_y): 
        y_prime = predict(xi, w)
        if yi != y_prime: 
            if yi == 0: 
                yi = -1
            w += (r * yi * xi)
    return w



def averaged_perception_update(train_x, train_y, a, r): 
    w = np.zeros(train_x.shape[1])
    for xi, yi in zip(train_x, train_y): 
        y_prime = predict(xi, w)
        if yi != y_prime: 
            if yi == 0: 
                yi = -1
            w += (r * yi * xi)
        a += w
    return a

def predict(xi, w): 
    yi = w @ xi
    if yi > 0: 
        return 1
    return 0

def perception(train_x, train_y, d, epochs, r):
    w = np.zeros(d)
    for epoch in range(epochs): 
        w = perception_update(train_x, train_y, w, r)
    return w 

def voted_perception_update(train_x, train_y, c, w, r): 
    weights = []
    d = w.shape[0]
    for xi, yi in zip(train_x, train_y): 
        y_prime = predict(xi, w)
        if yi != y_prime: 
            weights.append((w, c))
            if yi == 0: 
                yi = -1
            w_next = np.zeros(d)
            w_next = w + (r * yi * xi)
            w = w_next
            c = 1
        else: 
            c += 1
    return weights

def voted(train_x, train_y, d, epochs, r):
    w = np.zeros(d)
    m = np.zeros(d)
    c = 0
    all_weights = []
    for epoch in range(epochs): 
        weights = voted_perception_update(train_x, train_y, c, w, r)
        all_weights += weights
        w, c = weights[len(weights) -1]
    return all_weights

def averaged_perception(train_x, train_y, d, epochs, r):
    a = np.zeros(4)
    for epoch in range(epochs): 
        a = averaged_perception_update(train_x, train_y, a, r)
    return a

def evaluate_w(w, test_x, test_y): 
    correct = 0
    for xi, yi in zip(test_x, test_y): 
        if yi == predict(xi, w): 
            correct += 1
    return correct / (test_x).shape[0]

def voted_prediction(xi, weights): 
    sum = 0
    for w in weights:
        wi, ci = w
        yi = predict(xi, wi)
        if yi == 0: 
            yi = -1
        sum += (ci * yi)
    if sum > 0: 
        return 1
    return 0 

def report_results(w, test_x, test_y): 
    print("The weight vector is {} with accuracy of {} on test data".format(w, evaluate_w(w, test_x, test_y)))

def report_voted_results(weights, test_x, test_y): 
    correct = 0
    for xi, yi in zip(test_x, test_y): 
        yprime = voted_prediction(xi, weights)
        if yprime == yi:
            correct +=1
    return correct / (test_x).shape[0]


    

if __name__ == "__main__":
    train_data_filename = "datasets/bank-note/train.csv"
    test_data_filename = "datasets/bank-note/test.csv"

    test_data = np.genfromtxt(test_data_filename, delimiter=',', dtype=float) 
    train_data = np.genfromtxt(train_data_filename, delimiter=',', dtype=float) 

    train_x = train_data[:, :4]
    train_y = train_data[:, 4]
    d = train_x.shape[1]
    perception_epoch = 10
    perception_r = 1

    test_x = test_data[:, :4]
    test_y = test_data[:, 4]

    w = perception(train_x, train_y, d, perception_epoch, 1)
    report_results(w, test_x, test_y)

    weights = voted(train_x, train_y, d, perception_epoch, 1)
    test_error = report_voted_results(weights, test_x, test_y)
    print("Voted perception has test error {}".format(test_error))

    a = averaged_perception(train_x, train_y, d, perception_epoch, 1)
    report_results(a, test_x, test_y)


