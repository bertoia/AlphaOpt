"""
Custom objective GP model, time GP model, acquisition function, evaluator function
goes here
"""
import GPyOpt
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from GPyOpt.core.evaluators.base import EvaluatorBase
from numpy.random import beta
from GPyOpt.core.task.cost import CostModel
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
import math
import numpy as np

"""
Objective GP Model
"""



"""
Time GP Model
Re-implement GPyOpt.core.task.cost.CostModel
"""



"""
Acquisition Function
"""  
#TODO: Still need to figure out correct sign for acquisition functions
class EIXplore(AcquisitionBase):
    """
    Usage: Cycle is a parameter deciding how often to explore. Cycle = 2 implies
    alternate between exploration and exploitation. Cycle = 3 implies explore
    once every 3 evaluations.
    """
    analytical_gradient_prediction = False
    
    jitter = 0
    
    explore = 0
    cycle = 3
    
    prev = None

    def __init__(self, model, space, optimizer, cost_withGradients=None, jitter=0.01, cycle=3):
        super(EIXplore, self).__init__(model, space, optimizer)

        self.jitter = jitter
        self.cycle = cycle
        if cost_withGradients == None:
             self.cost_withGradients = constant_cost_withGradients
        else:
             self.cost_withGradients = cost_withGradients 


    def _compute_acq(self,x):
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)
        h = 0.5 * np.log(2*math.pi*math.e*np.square(s))
        if self.prev != None and abs(self.prev-fmin) < 1:
            self.prev = fmin
            return h
        
        self.prev = fmin
        f_acqu_x = h if (self.explore % self.cycle) == 0 else (fmin - m + self.jitter) * Phi + s * phi
        self.explore += 1
        self.explore %= self.cycle
        return f_acqu_x

class EntropyWeightedEI(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None):
        super(EntropyWeightedEI, self).__init__(model, space, optimizer)
        
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)

        if cost_withGradients == None:
             self.cost_withGradients = constant_cost_withGradients
        else:
             self.cost_withGradients = cost_withGradients 


    def _compute_acq(self,x):
        m, s = self.model.predict(x)
        acqu_x = self.EI.acquisition_function(x)
        
        h = 0.5 * np.log(2*math.pi*math.e*np.square(s))
        for i in range (acqu_x.shape[0]):
            acqu_x[i] += h[i]
        return acqu_x

# Meant to be used with posterior sampling
class EntropyExplore(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None):
        super(EntropyExplore, self).__init__(model, space, optimizer)
        
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)

        if cost_withGradients == None:
             self.cost_withGradients = constant_cost_withGradients
        else:
             self.cost_withGradients = cost_withGradients 


    def _compute_acq(self,x):
        m, s = self.model.predict(x)
        h = 0.5 * np.log(2*math.pi*math.e*np.square(s))
        return h

class PITarget(AcquisitionBase):
    """
    Usage: Target is the target output value (optimum of the black-box function)
    that we want to hit. This is usually unknown except for test functions.
    However, in our experiments, the target would be the accuracy, so depending
    on the units, target = 100 or target = 1 (default)
    """
    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.2, target=None):
        super(PIThreshold, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.target = 1 if target is None else target

    def _compute_acq(self, x):
        m, s = self.model.predict(x)
        fmin = self.target if self.target is not None else self.model.get_fmin()
        _, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu = Phi
        self.jitter *= .5
        return f_acqu

    def _compute_acq_withGradients(self, x):
        fmin = self.target if self.target is not None else self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi        
        df_acqu = -(phi/s)* (dmdx + dsdx * u)

        self.jitter *= .5
        return f_acqu, df_acqu

class MultiAcquisitions(EvaluatorBase):
    """
    Usage: Pass in a list of acquisition functions that you want to be evaluated
    across all cores.

    See Parallel Modular BO.ipynb for an example.
    """
    def __init__(self, *args):
        self.acquisitions = args

    def compute_batch(self):
        X_batch = self.acquisitions[0].optimize()
        for i in range(1, len(self.acquisitions)):
            X_batch = np.vstack((X_batch, self.acquisitions[i].optimize()))
        return X_batch
