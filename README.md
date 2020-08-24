# Mixed Variable Multi-Objective Optimisation (MVMOO)
## Details
MVMOO utilises [GPflow 2.0+](https://gpflow.org) to perform multi-objective optimisation for mixed variable systems. The algorithm utilises the gower distance metric to enable Bayesian multi-objective optimisation methods to be used on mixed variable systems.

## Installation
To install it is recommended you follow the installation instructions for GPflow 2.0+ and [Tensorflow 2.1+](https://www.tensorflow.org/install), once Tensorflow and GPflow have been successfully installed it is then recommended to install the remaining packages (see **requirements.txt**)

A future release utilising PyPI is planned. In the meantime you can clone the repository using git

```git
git clone https://github.com/jmanson377/MVMOO
```
## Usage
An example on how to use the optimisation algorithm is given below. This is for the optimisation of a mixed variable version of the VLMOP2 test problem

```python
import numpy as np
import matplotlib.pyplot as plt
from multi_mixed_optimizer import MVMOO

def discretevlmop2(s):
    x = s[:,:2]
    d = s[:, 2]
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = np.zeros((np.shape(s)[0],1))
    y2 = np.zeros((np.shape(s)[0],1))
    for i in range(np.shape(s)[0]):
        if d[i] == 1:
            y1[i] = 1 - np.exp(-1 * part1[i])
            y2[i] = 1 - np.exp(-1 * part2[i])
        else:
            y1[i] = 1.25 - np.exp(-1 * part1[i])
            y2[i] = 0.75 - np.exp(-1 * part2[i])
            
    return np.hstack((y1, y2))
```
Once the packages have been imported you can perform the multi-objective optimisation as follows, in the current form the bounds for discrete variables must be place at the end of the bounds array
```python
optimiser = MVMOO(input_dim=3, num_qual=1, num_obj=2, bounds=np.array([[-2,-2,1],[2,2,2]]))
Xtest = optimiser.sample_design(samples=100000, design='random')
Ytest = discretevlmop2(Xtest)
Yfront = optimiser.paretofront(Ytest)
sortedind = np.argsort(Yfront[:,0])
Ysorted = Yfront[sortedind,:]

plt.scatter(Ysorted[:,0],Ysorted[:,1],label='Pareto Front',s=10)
plt.xlabel(r'$f_1$')
plt.ylabel(r'$f_2$')
plt.title('Mixed Variable vlmop2 Pareto Front')
plt.show()

Xstore = []
Ystore = []
for k in range(1):
    X = optimiser.sample_design(samples=5, design='lhc')
    Y = discretevlmop2(X)
    for i in range(30):
        start = time.time()
        xmax, _ = optimiser.multinextcondition(X,Y)
        end = time.time()
        print("Time elapsed get next condition: " + str(end - start) + " seconds.")
        ysample = discretevlmop2(xmax.reshape((1,3)))
        print(xmax)
        print(ysample)
        X = np.concatenate((X,xmax))
        Y = np.concatenate((Y,ysample))
        if i == 0:
            Xiter = xmax
            yiter = ysample
        else:
            Xiter = np.concatenate((Xiter,xmax))
            yiter = np.concatenate((yiter,ysample))
    Xstore.append(X)
    Ystore.append(Y)
plt.scatter(Ysorted[:,0],Ysorted[:,1],label='Pareto Front',s=2)
plt.scatter(Y[:10,0],Ysorted[:10,1],label='Initial',s=10)
plt.scatter(Y[10:,0],Y[10:,1],label='Algorithm',s=30)
plt.show()
```
