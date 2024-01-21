import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train, y_train, X_test, y_test = X[:50000], y[:50000], X[50000:], y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    unique_classes = np.unique(y)
    class_priors = [np.sum(y == c)/(y.size) for c in unique_classes]
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    unique_classes = np.unique(y)
    unique_x = np.unique(X)
    destinations = X.shape[1]

    indexs = X.shape[0]
    
    P=[]
    P =  [
            [
                [
                    sum(
                        X[i, d] == x and y[i] == c
                        for i in range(indexs)
                    ) / np.sum(y == c)
                    for d in range(destinations)
                ]
                for c in unique_classes
            ]
            for x in unique_x
        ]
    pAcd, pCcd, pGcd, pTcd = P[0], P[1], P[2], P[3]
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    P = [pAcd, pCcd, pGcd, pTcd]
    unique_x = np.unique(X)
    destinations = X.shape[1]
    indexs = X.shape[0]
    
    score_values=[]
    for i in range(indexs):
        arr=[]
        for c in range(0,2):
            score=1
            for d in range(destinations):
                score = score * P[np.where(unique_x == X[i][d])[0][0]][c][d]
            score = np.log(score) + np.log(class_priors[c])
            arr.append(score)
        score_values.append(arr)
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    predicted_classes = np.argmax(scores, axis=1)
    predicted_classes= [p + 1 for p in predicted_classes]
    
    num_classes = len(np.unique(y_truth))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_class, predicted_class in zip(y_truth, predicted_classes):
        confusion_matrix[predicted_class - 1, true_class - 1] += 1
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
