import GPy
from experiments.experiment import run_standard_bo
import experiments.svm.svm_components as svm
from AlphaOpt.components import EIXplore, EntropyWeightedEI,\
    EntropyExplore, PITarget, RandomAcquisition
import numpy as np


cases = {
    "EIX_SVM": {"acquisition": EIXplore},
    "EntWI_SVM": {"acquisition": EntropyWeightedEI},
    "EntEX_SVM": {"acquisition": EntropyExplore},
    "PIT_SVM": {"acquisition": PITarget},
    "RAND_SVM": {"acquisition": RandomAcquisition}
}

file_midfix = "_X_init2_STD_ITER10"

for name, dd in cases.items():
    if name != "RAND_SVM":
        dd["bo"] = run_standard_bo(space=svm.space,
                                   objective=svm.objective,
                                   acquisition=EIXplore,
                                   X_init=svm.X_init2,
                                   file_prefix=name+file_midfix,
                                   max_iter=10, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=GPy.kern.Matern52,
                                   cost=True)
    else:
        dd["bo"] = run_standard_bo(space=svm.space,
                                   objective=svm.objective,
                                   acquisition=RandomAcquisition,
                                   X_init=svm.X_init_k(12),
                                   file_prefix=name+"_X_init2_STD_ITER10",
                                   max_iter=None, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=GPy.kern.Matern52,
                                   cost=True)

# import experiments.experiment.save_result
# save_result(cases)
for name, dd in cases.items():
    try:
        bo = dd["bo"]
        print(name)
        print('Value at minimum: ' + str(min(bo.Y)).strip('[]'))
        print('Best found minimum location: ' + str(bo.X[np.argmin(bo.Y), :]).strip('[]'))
        bo.plot_convergence(r"results\\" + name + file_midfix)
    except:
        continue
