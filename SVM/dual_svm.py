import numpy as np
import scipy
from helpers import *
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import math

def perform_dual_svm(train_data_filename, test_data_filename):
    train_data = np.genfromtxt(train_data_filename, delimiter=',', dtype=float) 
    test_data = np.genfromtxt(test_data_filename, delimiter=',', dtype=float) 
    
    process_labels(np, train_data)
    process_labels(np, test_data)

    test_data_linear = attach_ones(np, test_data)

    X, y = seperate_data(train_data)
    aCs = [100/873, 500/873, 700/873]
    bCs = aCs[:]
    blambdas = [0.1, 0.5, 1, 5, 100]
    #perform_linear(X, y, test_data_linear, aCs)
    #perform_nonlinear(X, y, test_data, bCs, blambdas)
    nonlinear_questionC(blambdas, bCs)

def recover_model(alphas, X, y): 
    w = np.zeros(X.shape[1])
    b = 0
    for xi, yi, ai in zip(X, y, alphas): 
        if ai != 0:
            w += ((ai * yi) * (xi))
            inner = w @ xi
            b = (yi - (inner))
    return w, b

def perform_nonlinear(X, y, test_data, Cs, blamdas): 
    question_c_info = []
    for lambda_ in blamdas:
        K_train = rbf_kernel(X, lambda_)
        testX, testY = seperate_data(test_data)
        for C in Cs:
            n_samples, n_features = X.shape
            K = np.dot(X, X.T)  # Kernel matrix

            # Objective function
            def objective(alpha):
                term1 = 0.5 * np.sum(
                    (alpha * y)[:, None] * (alpha * y)[None, :] * K
                )
                term2 = np.sum(alpha)
                return term1 - term2

            def equality_constraint(alpha):
                return np.dot(alpha, y)

            bounds = [(0, C) for _ in range(n_samples)]

            # Initial guess for alpha
            initial_alpha = np.zeros(n_samples)

            result = minimize(
            fun=objective,
            x0=initial_alpha,
            bounds=bounds,
            constraints={"type": "eq", "fun": equality_constraint},
            method="SLSQP"  
            )

            # Extract the solution
            alpha_opt = result.x
            # alpha_opt = np.genfromtxt('C:/Users/lujan/vsc/Fall2024/MachineL/MachineLearning_DecisionTree/SVM/out/nonlinear_alphas0.1145475372279496.out')
            np.savetxt('SVM/out/nonlinear_alphas{}andlambda{}.out'.format(C, lambda_), alpha_opt, delimiter=',') 
            
            support_indices = np.where((alpha_opt > 1e-5) & (alpha_opt < C))[0]
            b = np.mean(
                y[support_indices]
                - np.sum((alpha_opt * y)[:, None] * K_train[:, support_indices], axis=0)
            )
            
            # Training predictions
            def predict(X, X_train, alpha_opt, y_train, gamma, b):
                # Compute the RBF kernel between X and X_train
                # n = X.shape[0]
                # K = np.exp(-np.sum((X - X_train[:n,:]) ** 2, axis=2) / gamma)
                # # Predict using the SVM decision rule
                # return np.sign(np.sum((alpha_opt * y_train) * K, axis=1) + b)
                predictions = []
                for xj in X:
                    sum = 0
                    for xi, yi, ai in zip(X_train, y_train, alpha_opt):
                        expon =( np.linalg.norm(xi - xj) ** 2) / (-gamma)
                        sum += (ai * yi * math.exp(expon))
                    if sum > 0:
                        sum = 1
                    else:
                        sum = -1
                    predictions.append(sum)
                return np.array(predictions)
                    

            y_train_pred = predict(X, X, alpha_opt, y, lambda_, b)
            y_test_pred = predict(testX, X, alpha_opt, y, lambda_, b)

            print("done prediction")
            
            # Compute errors
            train_error = 1 - accuracy_score(y, y_train_pred)
            test_error = 1 - accuracy_score(testY, y_test_pred)

            print("Lambda: {}, C: {}, Train Error:{}, Test Error: {}".format(lambda_, C, train_error, test_error))
            question_c_info.append(alpha_opt)

def nonlinear_questionC(lambda_, C):
    i = 0
    mid_c = []
    for l in lambda_:
        i = 0
        for c in C:
            alphas = np.genfromtxt('C:/Users/lujan/vsc/Fall2024/MachineL/MachineLearning_DecisionTree/SVM/out/nonlinear_alphas{}andlambda{}.out'.format(c, l))
            #np.savetxt('SVM/out/, alpha_opt, delimiter=',') 
            # alphas = info[i]
            threshold = 1e-5
            nonzero_count = np.sum(alphas > threshold)
            print("For c = {} and lambda = {}, number of support vectors is {}".format(c, l, nonzero_count))
            if i == 1:
                mid_c.append(alphas)
            i += 1
            # this_c.append(alphas)
        # individual_cs.append(this_c)
    
    #checking the middle c
    # mid_c = individual_cs[1]
    for i in range(len(mid_c) - 1):
        current = mid_c[i]
        next = mid_c[i+1]
        num_similar = compare_vectors(current, next)
        
        print(f"Number of similar vectors between iteration {lambda_[i]} and {lambda_[i+1]}: {num_similar}")

# Define threshold

def compare_vectors(a1, a2):
    threshold = 1e-5
    count = 0
    for ai1, ai2 in zip(a1, a2):
        if ai1 > threshold and ai1 == ai2:
            count += 1
    return count



# Function to filter non-zero vectors based on the threshold
# def filter_non_zero_vectors(vectors, threshold):
#     return [v for v in vectors if np.linalg.norm(v) > threshold]


def rbf_kernel(X, lambda_):
    """Compute the RBF kernel matrix."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-np.linalg.norm(diff) ** 2 / lambda_)
    return K

def perform_linear(X, y, test_data, Cs): 
    for C in Cs:
        n_samples, n_features = X.shape
        K = np.dot(X, X.T)  # Kernel matrix

        # Objective function
        def objective(alpha):
            term1 = 0.5 * np.sum(
                (alpha * y)[:, None] * (alpha * y)[None, :] * K
            )
            term2 = np.sum(alpha)
            return term1 - term2

        def equality_constraint(alpha):
            return np.dot(alpha, y)

        bounds = [(0, C) for _ in range(n_samples)]

        # Initial guess for alpha
        initial_alpha = np.zeros(n_samples)

        result = minimize(
        fun=objective,
        x0=initial_alpha,
        bounds=bounds,
        constraints={"type": "eq", "fun": equality_constraint},
        method="SLSQP"  
        )

        # Extract the solution
        alpha_opt = result.x
        np.savetxt('SVM/out/alphas{}.out'.format(C), alpha_opt, delimiter=',') 
        w, b = recover_model(alpha_opt, X, y)
        print("For C = {}. We have w = {}, b = {}".format(C, w, b))

        report_results(np.insert(w, 0, b), test_data, "test")
    

if __name__ == "__main__":
    #use rate, use different c's

    train_data_filename = "datasets/bank-note/train.csv"
    test_data_filename = "datasets/bank-note/test.csv"
    perform_dual_svm(train_data_filename, test_data_filename)

   

    

