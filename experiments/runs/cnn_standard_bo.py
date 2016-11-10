import GPy
from experiments.experiment import run_standard_bo
import experiments.cnn.cnn_components as cnn
from AlphaOpt.components import EIXplore, EntropyWeightedEI,\
    EntropyExplore, PITarget, RandomAcquisition
import numpy as np
cases = {
    "EIX_CNN": {"acquisition": EIXplore},
    "EntWI_CNN": {"acquisition": EntropyWeightedEI},
    "EntEX_CNN": {"acquisition": EntropyExplore},
    "PIT_CNN": {"acquisition": PITarget},
    "RAND_CNN": {"acquisition": RandomAcquisition}
}

file_midfix = "_X_init2_STD_ITER10"

for name, dd in cases.items():
    if name != "RAND_CNN":
        dd["bo"] = run_standard_bo(space=cnn.space,
                                   objective=cnn.objective,
                                   acquisition=EIXplore,
                                   X_init=cnn.X_init2,
                                   file_prefix=name+"_X_init2_STD_ITER10",
                                   max_iter=10, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=GPy.kern.Matern52,
                                   cost=True)
    else:
        dd["bo"] = run_standard_bo(space=cnn.space,
                                   objective=cnn.objective,
                                   acquisition=RandomAcquisition,
                                   X_init=cnn.X_init_k(12),
                                   file_prefix=name+"_X_init2_STD_ITER10",
                                   max_iter=None, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=GPy.kern.Matern52,
                                   cost=True)

# import experiments.experiment.save_result
# save_result(cases)
for name, dd in cases.items():
    bo = dd["bo"]
    print(name)
    print('Value at minimum: ' + str(min(bo.Y)).strip('[]'))
    print('Best found minimum location: ' + str(bo.X[np.argmin(bo.Y), :]).strip('[]'))
    bo.plot_convergence(r"results\\" + name + file_midfix)
