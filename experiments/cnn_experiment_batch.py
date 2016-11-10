import experiments.experiment as exp
from experiments.cnn.cnn_components import cnn_accuracy_base
from experiments.cnn.cnn_components import domain as space
from AlphaOpt.components import CustomCostModel as CostModel

import GPyOpt
import GPy

objective_function = cnn_accuracy_base()
space = GPyOpt.Design_space(space=space)


objective = GPyOpt.core.task.SingleObjective(objective_function)
X_init = GPyOpt.util.stats.initial_design('random', space, 2)

# Bayesian Optimization Components
# GP models
model = GPyOpt.models.GPModel(GPy.kern.Matern52(input_dim=space.input_dim(), ARD=True),
                              optimize_restarts=5, verbose=False)
cost = CostModel(GPy.kern.Matern52(input_dim=space.input_dim(), ARD=True), 'evaluation_time')

# Decision models
acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
acquisition = GPyOpt.acquisitions.EI.AcquisitionEI(model, space, acquisition_optimizer)
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

#Combine everything
bo = GPyOpt.methods.ModularBayesianOptimization(model=model,
                                                space=space,
                                                objective=objective,
                                                acquisition=acquisition,
                                                evaluator=evaluator,
                                                X_init=X_init,
                                                cost=cost)

exp.run_bo_slice(bo, 5)