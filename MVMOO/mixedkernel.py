# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:54:11 2019

@author: pmjm

Currently working with GPflow >= 2.0, an example is given below, this should enable an update of the optimisation algorithm
"""

import gpflow as gpf
from gpflow.utilities import positive
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities.ops import square_distance

plt.style.use('seaborn-darkgrid')
###############################################################################
class MixedMatern52(gpf.kernels.Kernel):
    """
    Kernel function for a mixed discrete-continuous variable input domain

     @phdthesis{Halstrup:2016,
        title={Black-box optimization of mixed discrete-continuous optimization problems},
        url={https://d-nb.info/112468123X/34},
        doi={10.17877/DE290R-17800},
        school={TU Dortmund University},
        author={Halstrup, Momchil},
        year={2016}}
    """
    def __init__(self, input_dim=1, variance=1.0, lengthscales=1.0, 
                num_qual=0, **kwargs): 
        """
        :param variance: the (initial) value for the variance parameter
        :param lengthscale: the (initial) value for the lengthscale parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param num_qual: the number of qualitative variables to optimiser, these
            must be in the end columns of the input array
        :param kwargs: accepts `name` and `active_dims`, which is a list of
            length input_dim which controls which columns of X are used
        """
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
        super().__init__(**kwargs)
        self.variance = gpf.Parameter(variance, transform=positive())
        self.lengthscales = gpf.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)
        self.num_qual = num_qual
    
    def _scaled_square_dist(self, X, X2, lengthscales):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        This function can deal with leading dimensions in X and X2. 
        In the sample case, where X and X2 are both 2 dimensional, 
        for example, X is [N, D] and X2 is [M, D], then a tensor of shape 
        [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D] 
        then the output will be [N1, S1, N2, S2].
        """

        X = X / lengthscales

        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.linalg.adjoint(Xs)
            return dist

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2 = X2 / lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
        dist += _broadcasting_elementwise_op(tf.add, Xs, X2s)
        return dist
    
    def scaled_square_dist(self, X, X2):  # pragma: no cover
        return self._scaled_square_dist(X, X2, self.lengthscales)
    
    def gower_distance(self, X, X2):
        if self.num_qual == 0:
            return self.scaled_square_dist(X,X2)
        else:
            if X2 is None:
                Xqual = X[:,-self.num_qual:] # qualitative variables
                Xquant = X[:,:-self.num_qual] # quantitative variables
                distquant = self._scaled_square_dist(Xquant,None,self.lengthscales[:-self.num_qual]) # quantitative variable distances
                distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(Xqual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
                quallengthscales = self.lengthscales[-self.num_qual:]   # get qualitative variable length scales            
                for i in range(self.num_qual):
                    compqual = tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(Xqual[:,i],(-1,1))))
                    distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                    
                dist = distquant + distqual
                return dist #/ self.input_dim
            
            Xqual = X[:,-self.num_qual:] # qualitative variables
            Xquant = X[:,:-self.num_qual] # quantitative variables
            
            X2qual = X2[:,-self.num_qual:] # qualitative variables
            X2quant = X2[:,:-self.num_qual] # quantitative variables
            
            
            distquant = self._scaled_square_dist(Xquant,X2quant,self.lengthscales[:-self.num_qual]) # quantitative variable distances
            distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(X2qual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
            quallengthscales = self.lengthscales[-self.num_qual:]
            
            for i in range(self.num_qual):
                compqual =  tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(X2qual[:,i],(-1,1))))
                distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                
            dist = distquant + distqual

            return dist #/ self.input_dim
            
    def K_r(self, r):
        sqrt5 = np.sqrt(5.)
        return self.variance * (1.0 + sqrt5 * r + 5. / 3. * tf.square(r)) * tf.exp(-sqrt5 * r)  
    
    def K(self, X, X2=None,presliced=False):
        #if not presliced:
        #    X, X2 = self._slice(X, X2)
        return self.K_r(tf.sqrt(tf.maximum(self.gower_distance(X, X2), 1e-36)))

    def K_diag(self, X,presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

class MixedMatern32(gpf.kernels.Kernel):
    """
    Kernel function for a mixed discrete-continuous variable input domain

     @phdthesis{Halstrup:2016,
        title={Black-box optimization of mixed discrete-continuous optimization problems},
        url={https://d-nb.info/112468123X/34},
        doi={10.17877/DE290R-17800},
        school={TU Dortmund University},
        author={Halstrup, Momchil},
        year={2016}}
    """
    def __init__(self, input_dim=1, variance=1.0, lengthscales=1.0, 
                num_qual=0, **kwargs): 
        """
        :param variance: the (initial) value for the variance parameter
        :param lengthscale: the (initial) value for the lengthscale parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param num_qual: the number of qualitative variables to optimiser, these
            must be in the end columns of the input array
        :param kwargs: accepts `name` and `active_dims`, which is a list of
            length input_dim which controls which columns of X are used
        """
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
        super().__init__(**kwargs)
        self.variance = gpf.Parameter(variance, transform=positive())
        self.lengthscales = gpf.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)
        self.num_qual = num_qual
    
    def _scaled_square_dist(self, X, X2, lengthscales):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        This function can deal with leading dimensions in X and X2. 
        In the sample case, where X and X2 are both 2 dimensional, 
        for example, X is [N, D] and X2 is [M, D], then a tensor of shape 
        [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D] 
        then the output will be [N1, S1, N2, S2].
        """

        X = X / lengthscales

        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.linalg.adjoint(Xs)
            return dist

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2 = X2 / lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
        dist += _broadcasting_elementwise_op(tf.add, Xs, X2s)
        return dist
    
    def scaled_square_dist(self, X, X2):  # pragma: no cover
        return self._scaled_square_dist(X, X2,self.lengthscales)
    
    def gower_distance(self, X, X2):
        if self.num_qual == 0:
            return self.scaled_square_dist(X,X2)
        else:
            if X2 is None:
                Xqual = X[:,-self.num_qual:] # qualitative variables
                Xquant = X[:,:-self.num_qual] # quantitative variables
                distquant = self._scaled_square_dist(Xquant,None,self.lengthscales[:-self.num_qual]) # quantitative variable distances
                distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(Xqual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
                quallengthscales = self.lengthscales[-self.num_qual:]   # get qualitative variable length scales            
                for i in range(self.num_qual):
                    compqual = tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(Xqual[:,i],(-1,1))))
                    distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                    
                dist = distquant + distqual
                return dist #/ self.input_dim
            
            Xqual = X[:,-self.num_qual:] # qualitative variables
            Xquant = X[:,:-self.num_qual] # quantitative variables
            
            X2qual = X2[:,-self.num_qual:] # qualitative variables
            X2quant = X2[:,:-self.num_qual] # quantitative variables
            
            
            distquant = self._scaled_square_dist(Xquant,X2quant,self.lengthscales[:-self.num_qual]) # quantitative variable distances
            distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(X2qual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
            quallengthscales = self.lengthscales[-self.num_qual:]
            
            for i in range(self.num_qual):
                compqual =  tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(X2qual[:,i],(-1,1))))
                distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                
            dist = distquant + distqual

            return dist #/ self.input_dim
            
    def K_r(self, r):
        sqrt3 = np.sqrt(3.)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)  
    
    def K(self, X, X2=None,presliced=False):
        #if not presliced:
        #    X, X2 = self._slice(X, X2)
        return self.K_r(tf.sqrt(tf.maximum(self.gower_distance(X, X2), 1e-36)))

    def K_diag(self, X,presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

class MixedSqExp(gpf.kernels.Kernel):
    """
    Kernel function for a mixed discrete-continuous variable input domain

     @phdthesis{Halstrup:2016,
        title={Black-box optimization of mixed discrete-continuous optimization problems},
        url={https://d-nb.info/112468123X/34},
        doi={10.17877/DE290R-17800},
        school={TU Dortmund University},
        author={Halstrup, Momchil},
        year={2016}}
    """
    def __init__(self, input_dim=1, variance=1.0, lengthscales=1.0, 
                num_qual=0, **kwargs): 
        """
        :param variance: the (initial) value for the variance parameter
        :param lengthscale: the (initial) value for the lengthscale parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param num_qual: the number of qualitative variables to optimiser, these
            must be in the end columns of the input array
        :param kwargs: accepts `name` and `active_dims`, which is a list of
            length input_dim which controls which columns of X are used
        """
        for kwarg in kwargs:
            if kwarg not in {'name', 'active_dims'}:
                raise TypeError('Unknown keyword argument:', kwarg)
        super().__init__(**kwargs)
        self.variance = gpf.Parameter(variance, transform=positive())
        self.lengthscales = gpf.Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)
        self.num_qual = num_qual
    
    def _scaled_square_dist(self, X, X2, lengthscales):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        This function can deal with leading dimensions in X and X2. 
        In the sample case, where X and X2 are both 2 dimensional, 
        for example, X is [N, D] and X2 is [M, D], then a tensor of shape 
        [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D] 
        then the output will be [N1, S1, N2, S2].
        """

        X = X / lengthscales

        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.linalg.adjoint(Xs)
            return dist

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2 = X2 / lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
        dist += _broadcasting_elementwise_op(tf.add, Xs, X2s)
        return dist
    
    def scaled_square_dist(self, X, X2):  # pragma: no cover
        return self._scaled_square_dist(X, X2,self.lengthscales)
    
    def gower_distance(self, X, X2):
        if self.num_qual == 0:
            return self.scaled_square_dist(X,X2)
        else:
            if X2 is None:
                Xqual = X[:,-self.num_qual:] # qualitative variables
                Xquant = X[:,:-self.num_qual] # quantitative variables
                distquant = self._scaled_square_dist(Xquant,None,self.lengthscales[:-self.num_qual]) # quantitative variable distances
                distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(Xqual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
                quallengthscales = self.lengthscales[-self.num_qual:]   # get qualitative variable length scales            
                for i in range(self.num_qual):
                    compqual = tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(Xqual[:,i],(-1,1))))
                    distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                    
                dist = distquant + distqual
                return dist #/ self.input_dim
            
            Xqual = X[:,-self.num_qual:] # qualitative variables
            Xquant = X[:,:-self.num_qual] # quantitative variables
            
            X2qual = X2[:,-self.num_qual:] # qualitative variables
            X2quant = X2[:,:-self.num_qual] # quantitative variables
            
            
            distquant = self._scaled_square_dist(Xquant,X2quant,self.lengthscales[:-self.num_qual]) # quantitative variable distances
            distqual = tf.zeros((tf.shape(Xqual)[0],tf.shape(X2qual)[0]),dtype=tf.float64) # set up distance matrix for qualitative variables
            quallengthscales = self.lengthscales[-self.num_qual:]
            
            for i in range(self.num_qual):
                compqual =  tf.not_equal(tf.reshape(Xqual[:,i],(-1,1)),tf.transpose(tf.reshape(X2qual[:,i],(-1,1))))
                distqual += tf.divide(tf.cast(tf.where(compqual,tf.ones(tf.shape(compqual)),\
                                         tf.zeros(tf.shape(compqual))),dtype=tf.float64), quallengthscales[i])
                
            dist = distquant + distqual

            return dist #/ self.input_dim
            
    def K_r(self, r):
        return self.variance * tf.exp(-r)  
    
    def K(self, X, X2=None,presliced=False):
        #if not presliced:
        #    X, X2 = self._slice(X, X2)
        return self.K_r(tf.sqrt(tf.maximum(self.gower_distance(X, X2), 1e-36)))

    def K_diag(self, X,presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

###############################################################################   
def _broadcasting_elementwise_op(op, a, b):
    r"""
    Apply binary operation `op` to every pair in tensors `a` and `b`.
    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))

###############################################################################
    
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

###############################################################################

from gpflow.utilities import print_summary
import time

X = np.random.random((20,1))
Xqual = np.random.randint(1,3,(20,1))
Xcomb = np.concatenate((X,Xqual),1)

Y = ftrig(Xcomb)

k1 = MixedMatern32(input_dim=2,num_qual=1, lengthscales=[1,1])
#k2 = gpf.kernels.Constant(1)

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
mean, var = mixedmodel.predict_y(xx)
line, = plt.plot(xx[:,0], mean, lw=2)
_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
plt.plot(xx[:,0],ftrig(xx))
plt.scatter(Xcomb[(Xqual==1).reshape(-1),0],Y[(Xqual==1).reshape(-1)])
plt.show()

xx = np.concatenate((np.linspace(0, 1.1, 100).reshape(100, 1),2*np.ones((100,1))),1)
mean, var = mixedmodel.predict_y(xx)
line, = plt.plot(xx[:,0], mean, lw=2)
_ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
plt.plot(xx[:,0],ftrig(xx))
plt.scatter(Xcomb[(Xqual==2).reshape(-1),0],Y[(Xqual==2).reshape(-1)])
plt.show()





print_summary(mixedmodel)
#xx = np.concatenate((np.linspace(0, 1.1, 10000).reshape(10000, 1),2*np.ones((10000,1))),1)
#mean, var = mixedmodel.predict_y(xx)