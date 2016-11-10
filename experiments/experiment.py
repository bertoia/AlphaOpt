from AlphaOpt.components import RandomAcquisition
from AlphaOpt.components import CustomCostModel as CostModel
from GPyOpt.methods.modular_bayesian_optimization import \
    ModularBayesianOptimization
from AlphaOpt.components import MyModularBayesianOptimization

import GPyOpt
import GPy
import numpy as np


def run_standard_bo(space, objective, acquisition, X_init,
                    file_prefix,
                    max_iter=None, max_time=None,
                    model_kernel=GPy.kern.Matern52,
                    cost_kernel=GPy.kern.Matern52,
                    cost=True):
    """
    Covers standard BO with or without cost
    """
    model = GPyOpt.models.GPModel(
        kernel=model_kernel(input_dim=space.dimensionality, ARD=True),
        optimize_restarts=1,
        verbose=False)
    if cost:
        cost = CostModel(
            kernel=cost_kernel(input_dim=space.dimensionality, ARD=True),
            cost_withGradients='evaluation_time')

    aquisition = acquisition(model, space,
                             GPyOpt.optimization.AcquisitionOptimizer(space))
    evaluator = GPyOpt.core.evaluators.Sequential(aquisition)
    return run_modular_bo_experiment(model, space, objective, acquisition,
                                     evaluator,
                                     X_init, cost, max_iter, max_time,
                                     save_models_parameters=True,
                                     evaluations_file=file_prefix + "_evaluation.txt")


def run_modular_bo_experiment(model, space, objective, acquisition, evaluator,
                              X_init, cost, max_iter, max_time,
                              save_models_parameters=True,
                              report_file=None,
                              evaluations_file=None,
                              models_file=None):
    bo = ModularBayesianOptimization(model=model,
                                     space=space,
                                     objective=objective,
                                     acquisition=acquisition,
                                     evaluator=evaluator,
                                     X_init=X_init,
                                     cost=cost)

    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=1e-8,
                        verbosity=True,
                        save_models_parameters=save_models_parameters,
                        report_file=report_file,
                        evaluations_file=evaluations_file,
                        models_file=models_file)

    return bo


def run_modular_bo_experiment_slice(model, space, objective, acquisition,
                                    evaluator,
                                    X_init, cost, max_iter, max_time, prefix,
                                    save_models_parameters=True,
                                    report_file=None,
                                    evaluations_file=None,
                                    models_file=None):
    """"
    Slice view
    """

    bo = ModularBayesianOptimization(model=model,
                                     space=space,
                                     objective=objective,
                                     acquisition=acquisition,
                                     evaluator=evaluator,
                                     X_init=X_init,
                                     cost=cost)

    for i in range(max_iter):
        bo.run_optimization(max_iter=1, max_time=max_time, eps=1e-8,
                            verbosity=True,
                            save_models_parameters=save_models_parameters,
                            report_file=report_file,
                            evaluations_file=None,
                            models_file=models_file)
        bo.plot_acquisition(prefix + str(i))

    bo.save_evaluations(evaluations_file=evaluations_file)
    return bo


def run_bo_slice(bo, max_iter, prefix=None, evaluations_file=None):
    """

    :param bo:
    :param max_iter:
    :param prefix:
    :param evaluations_file:
    :return:
    """
    filename = None
    for i in range(max_iter):
        bo.run_optimization(max_iter=1)
        if prefix:
            filename = prefix + "_acq" + str(i)
        print(bo.model.get_model_parameters())
        print(bo.model.get_model_parameters_names())
        bo.plot_acquisition(filename)

    if evaluations_file:
        bo.save_evaluations(evaluations_file=evaluations_file)


def run_random_benchmark(objective, space, max_time, max_iter, X_init):
    dim = len(space)
    optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    model = GPyOpt.models.GPModel(
        kernel=GPy.kern.Matern52(input_dim=dim, ARD=True),
        optimize_restarts=5,
        verbose=False)

    cost = CostModel(kernel=GPy.kern.Matern52(input_dim=dim, ARD=True),
                     cost_withGradients='evaluation_time')
    acquisition = RandomAcquisition(model, space, optimizer)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    return run_modular_bo_experiment(model=model,
                                     space=space,
                                     objective=objective,
                                     acquisition=acquisition,
                                     evaluator=evaluator,
                                     X_init=X_init,
                                     cost=cost,
                                     max_time=max_time,
                                     max_iter=max_iter)


def save_result(cases):
    for name, dd in cases.items():
        bo = dd["bo"]
        print(name)
        print('Value at minimum: ' + str(min(bo.Y)).strip('[]'))
        print('Best found minimum location: ' + str(
            bo.X[np.argmin(bo.Y), :]).strip('[]'))
        bo.plot_convergence(name)


# from GPyOpt.plotting.plots_bo import plot_acquisition
# model_parameter_values = bo.model_parameters_iterations
# model_parameter_names = bo.model.get_model_parameters_names()
# X = bo.X
# Y = bo.Y
# def printer(model_parameters_iterations, model, cost, X, Y, file_prefix):
# update model
# print and save
# i=len(bo.X)-len(model_parameter_values)
# for parameter_value in model_parameter_values:
#     model = bo.model.copy()
#     model.model[:] = parameter_value
#     model.model.set_XY(X[:i],Y[:i])
#     plot_acquisition(bo.acquisition.space.get_bounds(),
#                      model.model.X.shape[1],
#                      model.model,
#                      model.model.X,
#                      model.model.Y,
#                      bo.acquisition.acquisition_function,
#                      X[i],)
#     i=i+1
#     if i==len(model_parameter_values):
#         break
