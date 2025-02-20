import numpy as np
import matplotlib.pyplot as plt

# Load data

t = np.loadtxt('/path/to/data/t')
x = np.loadtxt('/path/to/data/x')
y = np.loadtxt('/path/to/data/y')

# Extract only the relevant columns (ignore first column, index)
x_fixed = x[:, 1:]  # Take remaining columns
y_fixed = y[:, 1].reshape(-1, 1)  # Take the second column and ensure it is a column vector

def train_adaline(xin, yd, eta, tol, maxepocas, par):

    N, n = xin.shape  # Number of samples (rows) and features (columns)

    if par == 1:
        wt = np.random.rand(n + 1, 1) - 0.5  # Include bias
        xin = np.hstack((np.ones((N, 1)), xin))  # Add bias column
    else:
        wt = np.random.rand(n, 1) - 0.5

    nepocas = 0
    eepoca = tol + 1
    evec = np.zeros(maxepocas)  # Error vector

    while nepocas < maxepocas and eepoca > tol:
        ei2 = 0
        xseq = np.random.permutation(N)  # Shuffle the order of the samples

        for i in xseq:
            xvec = xin[i, :].reshape(-1, 1)  # Ensure column vector
            yhati = np.dot(xvec.T, wt)  # Compute output
            ei = yd[i] - yhati  # Compute error
            dw = eta * ei * xvec  # Compute weight update
            wt += dw  # Update weights
            ei2 += ei ** 2  # Accumulate squared error

        evec[nepocas] = float(ei2) / N  # Ensure ei2 is a scalar before assignment
        eepoca = evec[nepocas]  # Update stopping condition
        nepocas += 1

    return wt, evec[:nepocas]  # Return final weights and error vector

# Train the Adaline model with the corrected data
wt, err = train_adaline(x_fixed, y_fixed, eta=0.01, tol=0.01, maxepocas=50, par=1)