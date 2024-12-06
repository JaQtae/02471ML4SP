#%%
import numpy as np
a = 0.6
sigma_eta = 0.8 # Variance of the signal noise
sigma_epsilon = 1 # Noise variance
sigma_s = np.sqrt(sigma_eta**2/(1-a**2)) # Variance of the signal itself
Gamma_ss = np.array([[1, a, a**2], [a, 1, a], [a**2, a, 1]]) * sigma_s**2 # Covariance matrix of the signal
Gamma_epsilon = np.eye(3) * np.square(sigma_epsilon) # Covariance matrix of the noise
Gamma_l = Gamma_ss+Gamma_epsilon
gamma_ss = np.array([[1], [a], [a**2]]) * sigma_s**2 # Cross-correlation vector
w = np.linalg.solve(Gamma_l, gamma_ss) # Solve systems of linear equations Ax = B for x
print(w) 

# %% 3.2.7
# Minimum MSE = variance of desired signal - projection of desired signal onto filter
MMSE3 = sigma_s**2 - gamma_ss.T @ w
print(MMSE3)

# %%
