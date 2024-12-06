#%% paramters
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

theta_t = np.array([[1], [2], [0.5]]) # True theta
noiselevel = 0.75   # Standard deviation of Gaussian noise on data
d = len(theta_t)    # Number of dimensions
Nmin = 4            # Minimal training set size
Nmax = 20           # Maximal training set size
Ntest = 10000       # Size of test set 
repetitions = 10    # number of repetitions

# Make statistical sample of test errors for different N 
# initilise arrays for test and train errors
np.random.seed(42)

test1 = np.empty((repetitions, Nmax-Nmin+1))
test2 = np.empty((repetitions, Nmax-Nmin+1))
train1 = np.empty((repetitions, Nmax-Nmin+1))
train2 = np.empty((repetitions, Nmax-Nmin+1))
Ns = (Nmax-Nmin+1)*[None]

print("Number of repetitions is ", str(repetitions))
for j in range(repetitions):
    # print("Repetition ", str(j+1), " of ", str(repetitions), " repetitions")
    # test data
    # d-dimensional model data set
    X1test  = np.random.randn(Ntest, d) 
    X1test[:,0] = 1
    Ttest = X1test @ theta_t
    noisetest = np.random.randn(Ntest, 1) * noiselevel
    
    Ttest = Ttest + noisetest
    # Small model (d-1) dimensional
    X2test=X1test[:,0:(d-1)]
    
    # training data
    # d-dimensional model data set
    XX1=np.random.randn(Nmax,d) 
    XX1[:,0] = 1
    TT = XX1 @ theta_t
    noise = np.random.randn(Nmax,1) * noiselevel
    TT = TT + noise
    # Small model (d-1) dimensional
    XX2=XX1[:,0:(d-1)]  
    
    
    ###################################################
    # X1 is the full model with d params => More flexible, but overfit more easily
    # X2 is the small model with d-1 params => Potentially more robust if N isnt large, 
    # but might underfit if N is large
    
    n=0  # counter

    for N in range(Nmin, Nmax+1):
        ###################################################
        # Pick the first N  input vectors in 
        ###################################################
        X1 = XX1[:N, :]
        X2 = XX2[:N, :]
        
        # and the corresponding targets
        T=TT[:N]
        
        # Find optimal weights for the two models
        theta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ T # complete this line
        ### We solve the normal equations to find the optimal weights ###
        theta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ T # complete this line

        # compute training set predictions
        Y1 = X1 @ theta1
        Y2 = X2 @ theta2
        
        # compute training error
        ### MSE training set ###
        ### Normalized to be per example ###
        err1 = np.mean((Y1 - T) ** 2) / N # complete this line
        err2 = np.mean((Y1 - T) ** 2) / N # complete this line
        
        # compute test set predictions
        Y1test = X1test @ theta1
        Y2test = X2test @ theta2
        ### MSE test set ###
        ### Normalized to be per example ###
        err1test = np.mean((Y1test - Ttest) ** 2) / Ntest # complete this line
        err2test = np.mean((Y2test - Ttest) ** 2) / Ntest # complete this line
              
        # save the results for later 
        test1[j,n]=err1test
        test2[j,n]=err2test
        train1[j,n]=err1
        train2[j,n]=err2
        Ns[n] =N
        n=n+1
        
print("Repetition ", str(j+1), " of ", str(repetitions), " repetitions done.")

# Plot results 

fig = plt.figure(figsize=(12, 8))

plt.errorbar(Ns, np.mean(train1, axis=0), yerr= np.std(train1, axis=0)/np.sqrt(repetitions), label='train1', linestyle = ":", color="red")
plt.errorbar(Ns, np.mean(train2, axis=0), yerr= np.std(train2, axis=0)/np.sqrt(repetitions), label='train2', linestyle = ":", color="blue")
plt.errorbar(Ns, np.mean(test1, axis=0), yerr= np.std(test1, axis=0)/np.sqrt(repetitions), label='test1', color="red")
plt.errorbar(Ns, np.mean(test2, axis=0), yerr= np.std(test2, axis=0)/np.sqrt(repetitions), label='test2', color="blue")
plt.grid()
plt.xlabel("training set size")
plt.ylabel("mean square errors (test and training)")

plt.legend(loc='upper right')
plt.show()



### 2.1.6 ###
# The error for the per example as function of the size of the training sets for the two models
# are eerily similar. The error for the test set is higher than the training set.

### 2.1.7 ###
# Large training set values with the noise variance:
# The train and test error converges towards the noise variance.
# The models are limited by the noise in the data and cant achieve errors smaller than the noise.
# Larger N => better fit => lower error => closer to the noise variance