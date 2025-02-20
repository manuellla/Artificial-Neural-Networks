import numpy as np
import pandas as pd

def train_perceptron(xin, yd, eta, tol, maxepocas, par):
    N, n = xin.shape  # Number of samples (rows) and features (columns)

    # Initialize weights
    if par == 1:
        wt = np.random.rand(n + 1, 1) - 0.5  # Add bias weight
        xin = np.hstack((-np.ones((N, 1)), xin))  # Add bias (-1 as first column)
    else:
        wt = np.random.rand(n, 1) - 0.5

    nepocas = 0
    eepoca = tol + 1
    evec = np.zeros(maxepocas)  # Error vector

    while nepocas < maxepocas and eepoca > tol:
        ei2 = 0
        xseq = np.random.permutation(N)  # Shuffle the order of the samples

        for i in xseq:
            yhati = (np.dot(xin[i, :], wt) >= 0).astype(float)  # Activation function (step function)
            ei = yd[i] - yhati  # Error calculation
            dw = eta * ei * xin[i, :].reshape(-1, 1)  # Weight update
            wt += dw  # Update weights
            ei2 += ei**2  # Accumulate squared error

        evec[nepocas] = float(ei2) / N  # Compute mean squared error
        eepoca = evec[nepocas]  # Update stopping condition
        nepocas += 1

    return wt, evec[:nepocas]  # Return final weights and error vector

def y_perceptron(xvec, w, par):
    if par == 1:
        xvec = np.hstack((np.ones((xvec.shape[0], 1)), xvec))  # Add bias column

    u = np.dot(xvec, w)  # Compute weighted sum
    y = (u >= 0).astype(float)  # Apply step activation function (binary classification)

    return y

def accuracy(predicoes, verdadeiros):
    corretos = np.sum(predicoes == verdadeiros)  # Count correct predictions
    acuracia = corretos / len(verdadeiros)  # Compute accuracy
    return acuracia

def confusion_matrix(predicoes, verdadeiros):
    # Create a confusion matrix using Pandas
    matriz_confusao = pd.crosstab(pd.Series(verdadeiros.flatten(), name="T"),
                                  pd.Series(predicoes.flatten(), name="P"))

    print("\nConfusion Matrix:")
    print(matriz_confusao)
