import numpy as np

def report_margins(x_train, x_train_y, w): 
    w_norm = np.linalg.norm(w)
    min_margin = None
    for i, xi in enumerate(x_train): 
        magnitude = w @ xi - 4 
        dist = abs(x_train_y[i] * magnitude) / w_norm
        print("for xi={}, dist={}".format(i, dist))

        if min_margin is None or min_margin > dist:
            min_margin = dist
    print("The margin is {}.".format(min_margin))

def disjunction(x, k): 
    x_k = x[:k]
    x_2k = x[k: 2*k] 
    k_or = False
    for x_k_i in x_k: 
        k_or = k_or or (not x_k_i)
    k2_or = False
    for x_k_i in x_2k: 
        k2_or = k2_or or x_k_i
    if k_or or k2_or:
        return 1
    return 0

def translate(yi): 
    if yi >= 0: 
        return 1
    return 0 

def equation(): 
   
    d = 25
    x = binary_matrix(d)

    k = 0
    #bias = k - (1/2)
    while (k + 1) * 2 <  d: 
        k += 1
        bias = k - (1/2)
        print("k={}".format(k))

        w = create_w(d, k)
        for xi in x: 
            yi = xi @ w + bias
            yi = translate(yi)
            y_truth = disjunction(xi, k)
            if yi != y_truth: 
                print("result {}, predicted {}, expected {}".format(yi == y_truth, yi, y_truth))
        print("done with k ={}".format(k))

#received from chatgbt         
def binary_matrix(d):
    n = 2 ** d  # Calculate the number of rows
    matrix = []

    for i in range(n):
        # Convert each number to binary with 'd' digits and store it as a list of integers
        binary_representation = [int(bit) for bit in f"{i:0{d}b}"]
        matrix.append(binary_representation)

    return matrix

def create_w(d, k): 
    w = np.zeros(d)
    for i in range(0, k): 
        w[i] = -1
        w[i + k] = 1
    return w


def question1(): 
    w = np.array([2, 3])
    x_a = np.array([[1, 1, 1, 1], [1, 1, -1, 1], [1, 0, 0, -1], [1, -1, 3, 1]])
    x_train_a = x_a[:, 1:3]
    x_train_y_a = x_a[:, 3]

    x_b = np.array([[1, 1, 1, 1], [1, 1, -1, 1], [1, 0, 0, -1], [1, -1, 3, 1], [1, -1, -1, 1]])
    x_train_b = x_b[:, 1:3]
    x_train_y_b = x_b[:, 3]

    #1a 
    print("question 1a")
    report_margins(x_train_a, x_train_y_a, w)

    print("question 1b")
    report_margins(x_train_b, x_train_y_b, w)
        


if __name__ == "__main__":
    # question1()
    equation()
  