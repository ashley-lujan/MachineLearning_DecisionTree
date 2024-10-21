# from utils.helpful import saveDataSet
#use matplot
import math, random
import numpy as np
import matplotlib.pyplot as plt
error_threshold = math.pow(10, -6)
maxT = 10000

class gradReturn():
    def __init__(self, r, w, error, cost_values):
        self.learning_rate = r
        self.learned_vector = w
        self.error = error
        self.cost_values = cost_values

    def __repr__(self):
        return "Learning rate " + str(self.learning_rate) + ", learned weighted vector: " + w_str(self.learned_vector) + ", error: " + str(round(self.error, 4))

def w_str(w):
    result = "["
    for wi in w:
        result += str(round(wi, 2)) + ", "
    result = result[0:-2] + "]"
    return result

def saveDataSet(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            row = [1]
            for el in terms:
                row.append(float(el))
            result.append(row)
    return result

def testWith(w, filename): 
    testData = saveDataSet(filename)
    return totalError(w, testData)

def w_zero(length):
    w = []
    for i in range(length): 
        w.append(0)
    return w

def gradientDescent(w, r, threshold, dataset):
    t = 0
    prevW = [math.inf] * len(w)
    currentError = totalError(w, dataset)
    cost_values = [currentError]
    while belowThreshold(prevW, w, threshold):
        gradient = gradient_loss_vector(w, dataset)
        prevW = w
        w = updateW(w, gradient, r)
        t +=1

        currentError = totalError(w, dataset)
        cost_values.append(currentError)
    return gradReturn(r, w, cost_values[-1], cost_values)

def belowThreshold(w1, w2, threshold):
    if w1[0] == math.inf:
        return True
    return euclideanNorm(subtractWs(w1, w2)) > threshold

def subtractWs(w1, w2):
    w_sub = []
    for i, wi in enumerate(w1):
        w_sub.append(wi - w2[i])
    return w_sub


def euclideanNorm(w):
    norm = 0
    for wi in w:
        norm += (wi * wi)
    return norm


def randomSample(dataset): 
    m = len(dataset)
    i = random.randint(0, m-1)
    return [dataset[i]]

def stochasticDescent(w, r, threshold, dataset):
    t = 0
    prevW = [math.inf] * len(w)
    currentError = totalError(w, dataset)
    cost_values = [currentError]
    while belowThreshold(w, prevW, threshold):
        gradSample = randomSample(dataset)
        gradient = gradient_loss_vector(w, gradSample)
        prevW = w
        w = updateW(w, gradient, r)
        t +=1
        currentError = totalError(w, dataset)
        cost_values.append(currentError)
        if t > maxT:
            break 
    return gradReturn(r, w, cost_values[-1], cost_values)

def totalError(w, dataset): 
    total = 0
    for x in dataset: 
        error = calcError(x, w)
        total += (error * error)
    return total * (1/2)

#[]
def gradient_loss_vector(w, dataset):
    wnext = []
    for j, wj in enumerate(w): 
        error_sum = 0
        for row in dataset: 
            error = calcError(row, w)
            error_sum += (error * row[j])
        wnext.append(-error_sum)
    return wnext


def calcError(x, w): 
    yi = x[len(x) - 1]
    error = yi - predict(w, x)
    return error

def predict(w, x): 
    result = 0
    for i, wi in enumerate(w): 
        result += wi * x[i]
    return result

def updateW(w, grad, r): 
    result = []
    for i, wi in enumerate(w): 
        result.append(wi - (r * grad[i]))
    return result


def descent(dataSet, r_lst, title, descentFunc):
    testSet = saveDataSet('../datasets/concrete/test.csv')
    wlength = len(dataSet[0]) - 1

    w0 = w_zero(wlength)

    final_error = []
    rs = []

    for r in r_lst:
        print("\n\n r is ", r)
        gradInfo = descentFunc(w0, r, error_threshold, dataSet)
        cost_values = gradInfo.cost_values
        print("W: ", gradInfo.learned_vector)
        print("cost of test", gradient_loss_vector(gradInfo.learned_vector, testSet))
        print("total error in test: ", round(totalError(gradInfo.learned_vector, testSet), 2))
        x = range(0, len(cost_values), 1)
        # y = np.arange(cost_values)
        # print(y)
        plt.plot(x, cost_values)
        plt.xlabel('Iteration')
        plt.ylabel('Total Error')

        # Add a title
        plt.title(title + ' Total Error at Each Iteration for R = ' + str(r))
        plt.show()

        # fin_err = cost_values[-1]
        # if fin_err != math.inf:
        #     final_error.append(cost_values[-1])
        #     rs.append(r)
        # print(gradInfo)

    # plt.plot(rs, final_error)
    # plt.xlabel('Rs with no infinite error')
    # plt.ylabel('Error')
    # plt.show()

def getMatrices(dataSet):
    X = []
    Y = []
    for x in dataSet:
        X.append(x[0: -1])
        Y.append(x[-1])
    return (X, Y)

def analytical(dataSet):
    resulting = getMatrices(dataSet)
    X = np.array(resulting[0])
    Y = np.array(resulting[1])

    XXT_1 = np.linalg.inv(np.transpose(X) @ X)
    XXT_1X = XXT_1 @ np.transpose(X)
    # XtY = np.matmul(np.transpose(X), Y)
    # w = XXT_1X * Y
    # w = np.matmul(XXT_1X, Y)
    w = XXT_1X @ Y

    # print(w)
    print(w_str(w))

    return



def linearMain():
    dataSet = saveDataSet('../datasets/concrete/train.csv')
    def experimental():
        descent(dataSet, [0.01], 'Batch Gradient', gradientDescent)
        descent(dataSet, [0.25, 0.125, 0.01, 0.001, 0.0001], 'Stochastic Gradient', stochasticDescent)

    # print("Batch Gradient")
    # descent(dataSet, [.01], 'Batch Gradient', gradientDescent)
    # print("Stochastic Gradient")
    # descent(dataSet, [0.125, 0.01, 0.001], 'Stochastic Gradient', stochasticDescent)

    analytical(dataSet)



def question5():
    dataset = [[1, 1, -1, 2, 1], [1, 1, 1, 3, 4], [1, -1, 1, 0, -1], [1, 1, 2, -4, -2], [1, 3, -1, -1, 0]]
    analytical(dataset)

def question5d():
    dataset = [[1, 1, -1, 2, 1], [1, 1, 1, 3, 4], [1, -1, 1, 0, -1], [1, 1, 2, -4, -2], [1, 3, -1, -1, 0]]
    w = [0, 0, 0, 0, 0]
    r = 0.1
    for it, example in enumerate(dataset):
        w_t_1 = [0] * len(w)
        grad = [0] * len(w)
        error = example[-1] - predict(w, example)
        print(it, end=" & ")
        for i, wi in enumerate(w):
            grad[i] = (error) * example[i]
            w_t_1[i] = wi + r * grad[i]
        print(w_str(grad), end=" & ")
        print(w_str(w[1:]), end=" & ")
        print(round(w[0], 4), end=" \\\\ \\hline ")
        print("")

        w = w_t_1

def question5c():
    dataset = [[1, 1, -1, 2, 1], [1, 1, 1, 3, 4], [1, -1, 1, 0, -1], [1, 1, 2, -4, -2], [1, 3, -1, -1, 0]]
    w = [0, 0, 0, 0, 0]
    r = 0.1
    for it, example in enumerate(dataset):
        w_t_1 = [0] * len(w)
        grad = [0] * len(w)
        error = example[-1] - predict(w, example)
        print(it, end=" & ")
        for i, wi in enumerate(w):
            grad[i] = (error) * example[i]
            w_t_1[i] = wi + r * grad[i]
        print(w_str(grad), end=" & ")
        print(w_str(w[1:]), end=" & ")
        print(round(w[0], 4), end=" \\\\ \\hline ")
        print("")

        w = w_t_1








if __name__ == '__main__':
    # question5()
    linearMain()
    # linearMain([1])
    question5c()