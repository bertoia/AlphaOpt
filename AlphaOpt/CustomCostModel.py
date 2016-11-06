from GPyOpt.core.task.cost import CostModel
from GPyOpt.models import GPModel


class CustomCostModel(CostModel):
    def __init__(self, kernel, cost_withGradients):
        super(CustomCostModel,self).__init__()

        self.cost_type = cost_withGradients

        # --- Set-up evaluation cost
        if self.cost_type == None:
            self.cost_withGradients = CostModel.constant_cost_withGradients
            self.cost_type = 'Constant cost'

        elif self.cost_type == 'evaluation_time':
            self.cost_model = GPModel(kernel=kernel,exact_feval=False, normalize_Y=False, optimize_restarts=5)
            self.cost_withGradients = self._cost_gp_withGradients
            self.num_updates = 0
        else:
            self.cost_withGradients = cost_withGradients
            self.cost_type = 'Used defined cost'

