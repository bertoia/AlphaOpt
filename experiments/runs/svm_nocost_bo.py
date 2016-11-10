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

iter_size = 10
file_midfix = "_X2_NC_ITER"+str(iter_size)

for name, dd in cases.items():
    if name != "RAND_SVM":
        dd["bo"] = run_standard_bo(space=svm.space,
                                   objective=svm.objective,
                                   acquisition=EIXplore,
                                   X_init=svm.X_init2,
                                   file_prefix=name+file_midfix,
                                   max_iter=iter_size, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=None,
                                   cost=None)
    else:
        dd["bo"] = run_standard_bo(space=svm.space,
                                   objective=svm.objective,
                                   acquisition=RandomAcquisition,
                                   X_init=svm.X_init_k(iter_size+2),
                                   file_prefix=name+file_midfix,
                                   max_iter=None, max_time=None,
                                   model_kernel=GPy.kern.Matern52,
                                   cost_kernel=None,
                                   cost=None)

# import experiments.experiment.save_result
# save_result(cases)
for name, dd in cases.items():
    try:
        bo = dd["bo"]
        print(name)
        print('Value at minimum: ' + str(min(bo.Y)).strip('[]'))
        print('Best found minimum location: ' + str(bo.X[np.argmin(bo.Y), :]).strip('[]'))
        bo.plot_convergence(r"results\\" + name + file_midfix)
        bo.plot_acquisition(r"results\\" + name + file_midfix + "acq")
    except:
        continue