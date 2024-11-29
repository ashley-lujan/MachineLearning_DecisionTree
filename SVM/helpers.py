def process_labels(np, dataset):
    d = dataset.shape[1]
    labels = np.where(dataset[:, d - 1] == 0, -1, dataset[:, d-1])
    dataset[:, d-1] = labels

def seperate_data(data): 
    d = data.shape[1]
    x = data[:, :(d-1)]
    y = data[:, (d - 1)]
    return x, y

def report_results(w, data, d_type): 
    x, y = seperate_data(data)
    print("The weight vector is {} with accuracy of {} on {} data".format(w, evaluate_w(w, x, y), d_type))

def evaluate_w(w, test_x, test_y): 
    correct = 0
    for xi, yi in zip(test_x, test_y): 
        if yi == predict(xi, w): 
            correct += 1
    return correct / (test_x).shape[0]

def predict(xi, w): 
    yi = w @ xi
    if yi > 0: 
        return 1
    return -1

def attach_ones(np, data): 
    ones_column = np.ones((data.shape[0], 1))
    return np.hstack((ones_column, data))