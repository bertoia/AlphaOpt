"""
Custom objective GP model, time GP model, acquisition function, evaluator function
goes here
"""
import GPyOpt
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
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
class EIXplore(AcquisitionBase):   
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
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

class jitter_integrated_EI(AcquisitionBase):
    def __init__(self, model, space, optimizer=None, cost_withGradients=None,
                 par_a=1, par_b=1, num_samples=100):
        super(jitter_integrated_EI, self).__init__(model, space, optimizer)

        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        self.samples = beta(self.par_a, self.par_b, self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)

    def acquisition_function(self, x):
        acqu_x = np.zeros((x.shape[0], 1))
        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]
            acqu_x += self.EI.acquisition_function(x)
        return acqu_x / self.num_samples

    def acquisition_function_withGradients(self, x):
        acqu_x = np.zeros((x.shape[0], 1))
        acqu_x_grad = np.zeros(x.shape)

        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]
            acqu_x_sample, acqu_x_grad_sample = self.EI.acquisition_function_withGradients(
                x)
            acqu_x += acqu_x_sample
            acqu_x_grad += acqu_x_grad_sample
        return acqu_x / self.num_samples, acqu_x_grad / self.num_samples
