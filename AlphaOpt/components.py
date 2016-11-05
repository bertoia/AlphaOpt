"""
Custom objective GP model, time GP model, acquisition function, evaluator function
goes here
"""
import GPyOpt
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from numpy.random import beta
from GPyOpt.core.task.cost import CostModel

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
# Example
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
