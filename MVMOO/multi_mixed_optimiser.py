import numpy as np
from scipy.stats import norm
from .mixed_optimiser import MVO
from scipy.optimize import shgo, differential_evolution, dual_annealing
import scipy as stats

class MVMOO(MVO):
    """
    Multi variate mixed variable optimisation
    """
    def __init__(self, input_dim=1, num_qual=0, num_obj=2, bounds=None):
        """
        Initialisation of the class
        """
        super().__init__(input_dim=input_dim, num_qual=num_qual, bounds=bounds)

        self.num_obj = num_obj


    def generatemodels(self, X, Y, scale=True, variance=1.0):
        """
        Generate a list containing the models for each of the objectives
        """
        self.nsamples, nobj = np.shape(Y)
        models = []
        if scale is True:
            self.Yscaled = self.scaley(Y)
            self.Xscaled = self.scaleX(X)
            for i in range(nobj):
                self.fitmodel(X, self.Yscaled[:,i].reshape((-1,1)), variance=variance)
                models.append(self.model)
            return models
        for i in range(nobj):
            self.fitmodel(X, Y[:,i].reshape((-1,1)))
            models.append(self.model)
            return models

    def is_pareto_efficient(self, costs, return_mask = True):
        """
        Find the pareto-efficient points for minimisation problem
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def paretofront(self, Y):
        """
        Return an array of the pareto front for the system, set up for a minimising
        """
        ind = self.is_pareto_efficient(Y, return_mask=False)
        return Y[ind,:]

    def EIM_Hypervolume(self, X):
        """
        Calculate the expected improvment matrix for a candidate point

        @ARTICLE{7908974, 
            author={D. {Zhan} and Y. {Cheng} and J. {Liu}}, 
            journal={IEEE Transactions on Evolutionary Computation}, 
            title={Expected Improvement Matrix-Based Infill Criteria for Expensive Multiobjective Optimization}, 
            year={2017}, 
            volume={21}, 
            number={6}, 
            pages={956-975}, 
            doi={10.1109/TEVC.2017.2697503}, 
            ISSN={1089-778X}, 
            month={Dec}}
        """
        f = self.currentfront

        nfx = np.shape(f)[0]
    
        nobj = np.shape(f)[1]
    
        nx = np.shape(X)[0]
    
        r = 1.1 * np.ones((1, nobj))
        y = np.zeros((nx, 1))
    
        ulist = []
        varlist = []
    
        for iobj in range(nobj):
            u, var = self.models[iobj].predict_y(X)
            ulist.append(u)
            varlist.append(var)
            
        u = np.concatenate(ulist, axis=1)
        var = np.concatenate(varlist, axis=1)
        std = np.sqrt(np.maximum(0,var))

        u_matrix = np.reshape(u.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        s_matrix = np.reshape(std.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        f_matrix = f.reshape((nfx,nobj,1)) * np.ones((1,1,nx))
        Z_matrix = (f_matrix - u_matrix) / s_matrix
        EI_matrix = np.multiply((f_matrix - u_matrix), norm.cdf(Z_matrix)) + np.multiply(s_matrix, norm.pdf(Z_matrix))
        y = np.min(np.prod(r.reshape(1,2,1)  - f_matrix + EI_matrix, axis=1) - np.prod(r - f, axis=1).reshape((-1,1)),axis=0).reshape((-1,1))

        return y

    def EIM_Euclidean(self, X):
        """
        Calculate the expected improvment matrix for a candidate point

        @ARTICLE{7908974, 
            author={D. {Zhan} and Y. {Cheng} and J. {Liu}}, 
            journal={IEEE Transactions on Evolutionary Computation}, 
            title={Expected Improvement Matrix-Based Infill Criteria for Expensive Multiobjective Optimization}, 
            year={2017}, 
            volume={21}, 
            number={6}, 
            pages={956-975}, 
            doi={10.1109/TEVC.2017.2697503}, 
            ISSN={1089-778X}, 
            month={Dec}}
        """
        f = self.currentfront

        nfx = np.shape(f)[0]
    
        nobj = np.shape(f)[1]
    
        nx = np.shape(X)[0]

        y = np.zeros((nx, 1))
    
        ulist = []
        varlist = []
    
        for iobj in range(nobj):
            u, var = self.models[iobj].predict_f(X)
            ulist.append(u)
            varlist.append(var)
            
        u = np.concatenate(ulist, axis=1)
        var = np.concatenate(varlist, axis=1)
        std = np.sqrt(np.maximum(0,var))

        u_matrix = np.reshape(u.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        s_matrix = np.reshape(std.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        f_matrix = f.reshape((nfx,nobj,1)) * np.ones((1,1,nx))
        Z_matrix = (f_matrix - u_matrix) / s_matrix
        EI_matrix = np.multiply((f_matrix - u_matrix), norm.cdf(Z_matrix)) + np.multiply(s_matrix, norm.pdf(Z_matrix))
        y = np.min(np.sqrt(np.sum(EI_matrix**2,axis=1)),axis=0).reshape(-1,1)

        return y

    def CEIM_Hypervolume(self, X):
        """
        Calculate the expected improvment matrix for a candidate point, given constraints

        @ARTICLE{7908974, 
            author={D. {Zhan} and Y. {Cheng} and J. {Liu}}, 
            journal={IEEE Transactions on Evolutionary Computation}, 
            title={Expected Improvement Matrix-Based Infill Criteria for Expensive Multiobjective Optimization}, 
            year={2017}, 
            volume={21}, 
            number={6}, 
            pages={956-975}, 
            doi={10.1109/TEVC.2017.2697503}, 
            ISSN={1089-778X}, 
            month={Dec}}
        """
        f = self.currentfront
    
        nobj = np.shape(f)[1]
    
        nx = np.shape(X)[0]
    
        r = 1.1 * np.ones((1, nobj))
        y = np.zeros((nx, 1))
    
        ulist = []
        varlist = []
    
        for iobj in range(nobj):
            u, var = self.models[iobj].predict_y(X)
            ulist.append(u)
            varlist.append(var)
            
        u = np.concatenate(ulist, axis=1)
        var = np.concatenate(varlist, axis=1)
        std = np.sqrt(np.maximum(0,var))
    
        for ix in range(nx):
            Z = (f - u[ix,:]) / std[ix,:]
            EIM = np.multiply((f - u[ix,:]), norm.cdf(Z)) + np.multiply(std[ix,:], norm.pdf(Z))
            y[ix] = np.min(np.prod(r - f + EIM, axis=1) - np.prod(r - f, axis=1))
        
        # Constraints
        ncon = len(self.constrainedmodels)

        uconlist = []
        varconlist = []
    
        for iobj in range(ncon):
            ucon, varcon = self.constrainedmodels[iobj].predict_y(X)
            uconlist.append(ucon)
            varconlist.append(varcon)
            
        ucon = np.concatenate(uconlist, axis=1)
        varcon = np.concatenate(varconlist, axis=1)
        stdcon = np.sqrt(np.maximum(0,varcon))

        PoF = np.prod(norm.cdf((0 - ucon) / stdcon), axis=1).reshape(-1,1)

        return y * PoF

    def AEIM_Hypervolume(self, X):
        """
        Calculate the  adaptive expected improvment matrix for a candidate point

        Adaptive addition based on https://arxiv.org/pdf/1807.01279.pdf
        """
        f = self.currentfront
        c = self.contextual

        nfx = np.shape(f)[0]
    
        nobj = np.shape(f)[1]
    
        nx = np.shape(X)[0]
    
        r = 1.1 * np.ones((1, nobj))
        y = np.zeros((nx, 1))
    
        ulist = []
        varlist = []
    
        for iobj in range(nobj):
            u, var = self.models[iobj].predict_y(X)
            ulist.append(u)
            varlist.append(var)
            
        u = np.concatenate(ulist, axis=1)
        var = np.concatenate(varlist, axis=1)
        std = np.sqrt(np.maximum(0,var))

        u_matrix = np.reshape(u.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        s_matrix = np.reshape(std.T,(1,nobj,nx)) * np.ones((nfx,1,1))
        f_matrix = f.reshape((nfx,nobj,1)) * np.ones((1,1,nx))
        c_matrix = c.reshape((nfx,nobj,1)) * np.ones((1,1,nx))
        Z_matrix = (f_matrix - u_matrix - c_matrix) / s_matrix
        EI_matrix = np.multiply((f_matrix - u_matrix), norm.cdf(Z_matrix)) + np.multiply(s_matrix, norm.pdf(Z_matrix))
        y = np.min(np.prod(r.reshape(1,2,1)  - f_matrix + EI_matrix, axis=1) - np.prod(r - f, axis=1).reshape((-1,1)),axis=0).reshape((-1,1))
    
        #for ix in range(nx):
        #    Z = (f - u[ix,:] - c) / std[ix,:]
        #    EIM = np.multiply((f - u[ix,:]), norm.cdf(Z)) + np.multiply(std[ix,:], norm.pdf(Z))
        #    y[ix] = np.min(np.prod(r - f + EIM, axis=1) - np.prod(r - f, axis=1))
        
        return y

    def EIMoptimiserWrapper(self, Xcont, Xqual, constraints=False):

        X = np.concatenate((Xcont.reshape((1,-1)), Xqual.reshape((1,-1))), axis=1)

        if constraints is not False:
            return -self.CEIM_Hypervolume(X)

        return -self.EIM_Euclidean(X)

    def AEIMoptimiserWrapper(self, Xcont, Xqual):

        X = np.concatenate((Xcont.reshape((1,-1)), Xqual.reshape((1,-1))), axis=1)

        return -self.AEIM_Hypervolume(X)
        
            

    def EIMmixedoptimiser(self, constraints, algorithm='Random Local', values=None):
        """
        Optimise EI search whole domain
        """
        if algorithm == 'Random':
            Xsamples = self.sample_design(samples=10000, design='halton')

            if constraints is False:
                fvals = self.EIM_Hypervolume(Xsamples)
            else:
                fvals = self.CEIM_Hypervolume(Xsamples)

            fmax = np.amax(fvals)
            indymax = np.where(fvals == fmax)
            xmax = Xsamples[int(indymax[0][0]),:]
            if values is None:
                return fmax, xmax
            return fmax, xmax, fvals, Xsamples
        elif algorithm == 'Random Local':
            Xsamples = self.sample_design(samples=10000, design='halton')

            if constraints is False:
                fvals = self.EIM_Euclidean(Xsamples)
            else:
                fvals = self.CEIM_Hypervolume(Xsamples)

            fmax = np.amax(fvals)
            indymax = np.where(fvals == fmax)
            xmax = Xsamples[int(indymax[0][0]),:]
            qual = xmax[-self.num_qual:]

            bnd = list(self.bounds[:,:self.num_quant].T)
            bndlist = []

            for element in bnd:
                bndlist.append(tuple(element))

            result = stats.optimize.minimize(self.EIMoptimiserWrapper, xmax[:-self.num_qual].reshape(-1), args=(qual,constraints), bounds=bndlist,method='SLSQP')
            if values is None:
                
                return result.fun, np.concatenate((result.x, qual),axis=None)

            return fmax, xmax, fvals, Xsamples

        elif algorithm == 'SHGO':
            if self.num_qual < 1:
                bnd = list(self.bounds.T)
                bndlist = []

                for element in bnd:
                    bndlist.append(tuple(element))
            
                if constraints is False:
                    result = shgo(self.EIM_Hypervolume,bndlist, sampling_method='sobol', n=30, iters=2)
                
                else:
                    result = shgo(self.CEIM_Hypervolume,bndlist, sampling_method='sobol', n=30, iters=2)
                return result.x, result.fun
            else:
                sample = self.sample_design(samples=1, design='random')
                contbnd = list(self.bounds[:,:self.num_quant].T)
                contbndlist = []
                qual = sample[:,-self.num_qual:]

                for element in contbnd:
                    contbndlist.append(tuple(element))
                resXstore = []
                resFstore = []
                for i in range(np.shape(qual)[0]):
                    result = shgo(self.EIMoptimiserWrapper, contbndlist, args=(qual[i,:],constraints), sampling_method='sobol', n=30, iters=2)
                    resXstore.append(result.x)
                    resFstore.append(result.fun)

                # sort for each discrete combination and get best point
                ind = resFstore.index(min(resFstore))  
                xmax = np.concatenate((resXstore[ind],qual[ind,:]))
                fval = min(resFstore)      
                return fval, xmax
        else:
            if self.num_qual < 1:
                bnd = list(self.bounds.T)
                bndlist = []

                for element in bnd:
                    bndlist.append(tuple(element))
            
                if constraints is False:
                    result = shgo(self.EIM_Hypervolume,bndlist, sampling_method='simplicial', iters=3)
                
                else:
                    result = shgo(self.CEIM_Hypervolume,bndlist, sampling_method='simplicial', iters=3)
                return result.x, result.fun
            else:
                sample = self.sample_design(samples=1, design='random')
                contbnd = list(self.bounds[:,:self.num_quant].T)
                contbndlist = []
                qual = sample[:,-self.num_qual:]

                for element in contbnd:
                    contbndlist.append(tuple(element))
                resXstore = []
                resFstore = []
                for i in range(np.shape(qual)[0]):
                    result = shgo(self.EIMoptimiserWrapper, contbndlist, args=(qual[i,:],constraints), sampling_method='simplicial', iters=3)
                    resXstore.append(result.x)
                    resFstore.append(result.fun)

                # sort for each discrete combination and get best point
                ind = resFstore.index(min(resFstore))  
                xmax = np.concatenate((resXstore[ind],qual[ind,:]))
                fval = min(resFstore)      
                return fval, xmax        

    
    def AEIMmixedoptimiser(self, algorithm='Random', values=None):

        # Get estimate for mean variance of model using halton sampling
        X = self.sample_design(samples=10000, design='halton')

        varlist = []
        for iobj in range(self.num_obj):
            _ , var = self.models[iobj].predict_y(X)
            varlist.append(var)

        var = np.concatenate(varlist, axis=1)
        meanvar = np.mean(var)

        f = self.currentfront

        self.contextual = np.divide(meanvar, f)

        # Optimise acquisition

        if algorithm == 'Random':

            Xsamples = self.sample_design(samples=10000, design='halton')

            fvals = self.AEIM_Hypervolume(Xsamples)

            fmax = np.amax(fvals)
            indymax = np.where(fvals == fmax)
            xmax = Xsamples[int(indymax[0][0]),:]
            if values is None:
                return fmax, xmax
            return fmax, xmax, fvals, Xsamples

        elif algorithm == 'SHGO':
            if self.num_qual < 1:
                bnd = list(self.bounds.T)
                bndlist = []

                for element in bnd:
                    bndlist.append(tuple(element))
            
                result = shgo(self.AEIM_Hypervolume,bndlist, sampling_method='sobol', n=30, iters=2)
                
                return result.x, result.fun
            else:
                sample = self.sample_design(samples=1, design='random')
                contbnd = list(self.bounds[:,:self.num_quant].T)
                contbndlist = []
                qual = sample[:,-self.num_qual:]

                for element in contbnd:
                    contbndlist.append(tuple(element))
                resXstore = []
                resFstore = []
                for i in range(np.shape(qual)[0]):
                    result = shgo(self.AEIMoptimiserWrapper, contbndlist, args=(qual[i,:]), sampling_method='sobol', n=30, iters=2)
                    resXstore.append(result.x)
                    resFstore.append(result.fun)

                # sort for each discrete combination and get best point
                ind = resFstore.index(min(resFstore))  
                xmax = np.concatenate((resXstore[ind],qual[ind,:]))
                fval = min(resFstore)      
                return fval, xmax

        elif algorithm == 'DE':
            if self.num_qual < 1:
                bnd = list(self.bounds.T)
                bndlist = []

                for element in bnd:
                    bndlist.append(tuple(element))
            
                result = differential_evolution(self.AEIM_Hypervolume,bndlist)
                
                return result.x, result.fun
            else:
                sample = self.sample_design(samples=1, design='random')
                contbnd = list(self.bounds[:,:self.num_quant].T)
                contbndlist = []
                qual = sample[:,-self.num_qual:]

                for element in contbnd:
                    contbndlist.append(tuple(element))
                resXstore = []
                resFstore = []
                for i in range(np.shape(qual)[0]):
                    result = dual_annealing(self.AEIMoptimiserWrapper, contbndlist, args=(qual[i,:]))
                    resXstore.append(result.x)
                    resFstore.append(result.fun)

                # sort for each discrete combination and get best point
                ind = resFstore.index(min(resFstore))  
                xmax = np.concatenate((resXstore[ind],qual[ind,:]))
                fval = min(resFstore)      
                return fval, xmax
        

        return      

    def multinextcondition(self, X, Y, constraints=False, values=None):
        """
        Suggest the next condition for evaluation
        """
        if constraints is False:
            try:
                self.models = self.generatemodels(X, Y)
            except:
                print('Initial model optimisation failed, retrying with new initial value for variance')
                self.models = self.generatemodels(X, Y, variance=0.1)

            self.currentfront = self.paretofront(self.Yscaled)

            means = []
            for model in self.models:
                mean, _ = model.predict_y(self.sample_design(samples=2, design='halton'))
                means.append(mean.numpy())
            if np.any(means == np.nan):
                print("Retraining model with new starting variance")
                self.models = self.generatemodels(X, Y, variance=0.1)

            fmax, xmax = self.EIMmixedoptimiser(constraints, algorithm='Random Local')
            #fmax, xmax = self.AEIMmixedoptimiser(algorithm='Random')
            if values is None:
                return xmax.reshape(1,-1), fmax
 
        self.models = self.generatemodels(X,Y)
        self.currentfront = self.paretofront(self.Yscaled)
        self.constrainedmodels = self.generatemodels(X, constraints, scale=False)

        fmax, xmax = self.EIMmixedoptimiser(constraints, algorithm='Simplical')
        if values is None:
            return xmax.reshape(1,-1), fmax
        