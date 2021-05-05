import gpflow as gpf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from gpflow.utilities import print_summary
import time
from MVMOO.mixedkernel import MixedMatern32
from MVMOO import MVMOO

# Discrete VLMOP2 test problem multi-objective

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

# ftrig test problem single objective

def ftrig(s):
    x = s[:,0]
    d = s[:,1]
    f = np.zeros((np.shape(s)[0],1))
    for i in range(np.shape(s)[0]):
        if d[i] == 1:
            f[i] = np.sin(6*(x[i]**2 - 0.25)) + 1.
        else:
            f[i] = np.sin(x[i])*np.tan(x[i]) + 0.1
    return f

# Code to determine pareto front
def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

## Single objective kernel test

X = np.random.random_sample((20,1))
Xqual = np.random.randint(1,3,(20,1))
Xcomb = np.concatenate((X,Xqual),1)

Y = ftrig(Xcomb) #+ 0.1 * np.random.randn(20,1)

k1 = MixedMatern32(input_dim=2,num_qual=1, lengthscales=np.ones((np.shape(Xcomb)[1])),dist='manhattan')
k2 = MixedMatern32(input_dim=2,num_qual=1, lengthscales=np.ones((np.shape(Xcomb)[1])),dist='manhattan')

k = k1 #+ k2

print_summary(k)

start = time.time()
mixedmodel = gpf.models.GPR(data=(Xcomb, Y), kernel=k)
opt = gpf.optimizers.Scipy()
logs = opt.minimize(
    mixedmodel.training_loss,
    variables=mixedmodel.trainable_variables,compile=False,
    options=dict(disp=False, maxiter=200),step_callback=None,
)
end = time.time()
print(end-start)

xx = np.concatenate((np.linspace(0, 1.1, 100).reshape(100, 1),1*np.ones((100,1))),1)
mean, var = mixedmodel.predict_f(xx)
line, = plt.plot(xx[:,0], mean.numpy(), lw=2)
_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
plt.plot(xx[:,0],ftrig(xx))
plt.scatter(Xcomb[(Xqual==1).reshape(-1),0],Y[(Xqual==1).reshape(-1)])

k=k2

mixedmodel = gpf.models.GPR(data=(Xcomb, Y), kernel=k)
opt = gpf.optimizers.Scipy()
logs = opt.minimize(
    mixedmodel.training_loss,
    variables=mixedmodel.trainable_variables,compile=False,
    options=dict(disp=False, maxiter=200),step_callback=None,
)

xx = np.concatenate((np.linspace(0, 1.1, 100).reshape(100, 1),1*np.ones((100,1))),1)
mean, var = mixedmodel.predict_f(xx)
line, = plt.plot(xx[:,0], mean.numpy(), lw=2, color='purple')
_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)


plt.show()

xx = np.concatenate((np.linspace(0, 1.1, 100).reshape(100, 1),2*np.ones((100,1))),1)
mean, var = mixedmodel.predict_y(xx)
line, = plt.plot(xx[:,0], mean, lw=2)
_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
plt.plot(xx[:,0],ftrig(xx))
plt.scatter(Xcomb[(Xqual==2).reshape(-1),0],Y[(Xqual==2).reshape(-1)])
plt.show()

print_summary(mixedmodel)

## Multi-Objective test
optimiser = MVMOO(input_dim=3, num_qual=1, num_obj=2, bounds=np.array([[-2.,-2.,1.],[2.,2.,2.]]))
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
    i=0
    while X.shape[0] < 30:
        start = time.time()
        xmax, fval = optimiser.multinextcondition(X,Y, method='EIM',mode='euclidean')
        end = time.time()
        print(fval)
        print("Time elapsed to get next condition: " + str(end - start) + " seconds.")
        ysample = discretevlmop2(xmax.reshape((-1,3)))
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
