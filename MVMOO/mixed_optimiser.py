import gpflow as gpf
import numpy as np
from scipy.stats import norm
from pyDOE2 import lhs
from .mixedkernel import MixedMatern52, MixedMatern32, MixedSqExp
import sobol_seq

class MVO():
    '''
    Class for mixed varibale bayesian optimisation
    '''

    def __init__(self, input_dim=1, num_qual=0, bounds=None):
        '''
        Initialisation of the class
        '''
        self.input_dim = input_dim
        self.num_qual = num_qual
        self.num_quant = input_dim - num_qual

        if bounds is None:
            bounds = np.zeros((2,input_dim))
            bounds[1,:] = np.ones((1,input_dim))
            self.bounds = bounds
        else:
            self.bounds = bounds

    def halton(self, dim, n_sample):
        """Halton sequence.
        :param int dim: dimension
        :param int n_sample: number of samples.
        :return: sequence of Halton.
        :rtype: array_like (n_samples, n_features)
        """

        def primes_from_2_to(n):
            """Prime number from 2 to n.
            From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
            :param int n: sup bound with ``n >= 6``.
            :return: primes in 2 <= p < n.
            :rtype: list
            """
            sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
            for i in range(1, int(n ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[k * k // 3::2 * k] = False
                    sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
            return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

        def van_der_corput(n_sample, base=2):
            """Van der Corput sequence.
            :param int n_sample: number of element of the sequence.
            :param int base: base of the sequence.
            :return: sequence of Van der Corput.
            :rtype: list (n_samples,)
            """
            n_sample, base = int(n_sample), int(base)
            sequence = []
            for i in range(n_sample):
                n_th_number, denom = 0., 1.
                while i > 0:
                    i, remainder = divmod(i, base)
                    denom *= base
                    n_th_number += remainder / denom
                sequence.append(n_th_number)

            return sequence

        big_number = 10
        while 'Not enought primes':
            base = primes_from_2_to(big_number)[:dim]
            if len(base) == dim:
                break
            big_number += 1000

        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [van_der_corput(n_sample + 1, dim) for dim in base]
        sample = np.stack(sample, axis=-1)[1:]

        return sample

    def sample_design(self, samples=5, design='random'):
        """
        Suggest different samples based on a selected design
        """
        if design == 'random':
            Xquant = np.random.random_sample((samples,self.num_quant))
        elif design == 'halton':
            Xquant = self.halton(self.num_quant, samples)
        elif design == 'sobol':
            Xquant = sobol_seq.i4_sobol_generate(self.num_quant, samples)
        elif design == 'lhc':
            Xquant = lhs(self.num_quant, samples=samples, criterion='maximin')
        else:
            raise ValueError("Please select a valid design: random, halton, sobol or lhc")

        if self.num_qual == 0:
            return np.multiply(Xquant, (self.bounds[1,:self.num_quant]-self.bounds[0,:self.num_quant])) + self.bounds[0,:self.num_quant]

        qlist = []
        qualbounds = self.bounds[:,self.num_quant:]

        for i in range(self.num_qual):
            qlist.append(np.linspace(qualbounds[0,i],qualbounds[1,i],qualbounds[1,i]))

        Xqual = np.array(np.meshgrid(*qlist)).T.reshape(-1,self.num_qual)

        Xcombined = np.zeros((samples*np.shape(Xqual)[0],self.input_dim))
        
        for i in range(int(np.shape(Xqual)[0])):
            Xcombined[i*samples:(i+1)*samples,:self.num_quant] = \
                np.multiply(Xquant, (self.bounds[1,:self.num_quant]-self.bounds[0,:self.num_quant])) + self.bounds[0,:self.num_quant]

            Xcombined[i*samples:(i+1)*samples,self.num_quant:] = np.multiply(np.ones((samples,self.num_qual)),Xqual[i,:])
            
        return Xcombined
    
    def scaleX(self, X, mode='meanstd', store=True):
        '''
        Scale the input variables
        '''
        Xcont = X[:,:self.input_dim-self.num_qual]
        if self.num_qual == 0:
            Xqual = []
        else:
            Xqual = X[:,-self.num_qual:]

        if mode == 'meanstd':
            if store is True:
                self.xmu = np.mean(Xcont, axis=0)
                self.xstd = np.std(Xcont, axis=0)
                return np.concatenate(((Xcont - self.xmu) / self.xstd, Xqual),axis=1)
            else:
                xmu = np.mean(Xcont, axis=0)
                xstd = np.std(Xcont, axis=0)
                return np.concatenate(((Xcont - xmu) / xstd, Xqual),axis=1)
            

        elif mode == 'bounds':
            return (X - self.bounds[0,:self.num_quant]) / (self.bounds[1,:self.num_quant] - self.bounds[0,:self.num_quant])

        else:
            raise ValueError("Select either 'meanstd' or 'bounds' scaling")

    def scaley(self, y, mode='meanstd'):
        """
        Scale the output variables
        """
        if mode == 'meanstd':
            self.ymu = np.mean(y, axis=0)
            self.ystd = np.std(y, axis=0)
            return (y - self.ymu) / self.ystd
        else:
            raise ValueError('No other scaling currently implemented')

    def fitmodel(self, X, y, variance=1.0):
        '''
        Fit the mixed variable model
        '''
        k = MixedMatern32(variance = variance, lengthscales=np.ones((1,self.input_dim)).reshape(-1),num_qual=self.num_qual)
        self.model = gpf.models.GPR(data=(X, y), kernel=k)
        try:
            optimizer = gpf.optimizers.Scipy()
            logs = optimizer.minimize(
                self.model.training_loss,
                variables=self.model.trainable_variables,compile=True,
                options=dict(disp=False, maxiter=200),step_callback=None)
        except Exception as e:
            print("Warning: Unable to perform optimisation of model\n")
            print(e)
            optimizer = gpf.optimizers.Scipy()
            logs = optimizer.minimize(
                self.model.training_loss,
                variables=self.model.trainable_variables,compile=True,
                options=dict(disp=False, maxiter=200),step_callback=None)

    def prediction(self,X):
        """
        Wrapper for predict_y with variable scaling
        STILL NEED TO IMPLEMENT
        """
        if self.num_qual == 0:
            transformed = self.scaleX(X)
        else:
            transformedX = np.concatenate((self.scaleX(X[:,:self.num_quant]),X[:,:self.num_quant]),axis=1)

        raise NotImplementedError()

    def expected_improvement(self, X):
        '''
        Expected improvement for next candidate point

        @article{Jones:1998,
            title={Efficient global optimization of expensive black-box functions},
            author={Jones, Donald R and Schonlau, Matthias and Welch, William J},
            journal={Journal of Global optimization},
            volume={13},
            number={4},
            pages={455--492},
            year={1998},
            publisher={Springer}}
        '''
        mean, var = self.model.predict_y(X)
        std = np.sqrt(var)
         

        Z = np.divide((self.yminsample - mean), std)
        return np.multiply(self.yminsample - mean, norm.cdf(Z)) + np.multiply(std, norm.pdf(Z))

    def mixedEIoptimiser(self):
        '''
        Optimise EI searching the whole domain
        '''
        Xsamples = self.sample_design(samples=10000, design='halton')

        fvals = self.expected_improvement(Xsamples)

        fmax = np.amax(fvals)
        indymax = np.where(fvals == fmax)
        xmax = Xsamples[int(indymax[0]),:]
        return fmax, xmax, fvals, Xsamples

    def nextconditions(self,X,y,values=None):
        '''
        method to suggest next conditions, could add variable scaling in future
        '''
        self.X = X
        self.y = y

        if self.num_qual == 0:
            self.Xscaled = self.scaleX(X)
        else:
            self.Xscaled = np.concatenate((self.scaleX(X[:,:self.num_quant]),X[:,self.num_quant:]),axis=1)

        self.yscaled = self.scaley(y)
        self.ymin = np.amin(y)
        indymin = np.where(y == self.ymin)
        self.xmin = X[int(indymin[0]),:].reshape(1,self.input_dim)

        self.fitmodel(X,y)
        self.yminsample, _ = self.model.predict_y(self.xmin)
        fmax, xmax, fvals, Xsamples = self.mixedEIoptimiser()
        if values is None:
            return  xmax.reshape(1,-1), fmax
        return xmax, fmax, fvals, Xsamples
        
    def optimisefunction(self, func, maxits, Xinitial, yinitial):
        """
        Optimise a user provided function, function needs to accept an array input
        """
        X = Xinitial
        y = yinitial
        for i in range(maxits):
            xmax, _ = self.nextconditions(X,y)
            ysample = func(xmax)
            X = np.concatenate((X,xmax))
            y = np.concatenate((y,ysample))
            if i == 0:
                Xiter = xmax
                yiter = ysample
            else:
                Xiter = np.concatenate((Xiter,xmax))
                yiter = np.concatenate((yiter,ysample))

        ymin = np.amin(y)
        indymin = np.where(y == ymin)
        xmin = X[int(indymin[0]),:]
        return xmin, ymin, Xiter, yiter