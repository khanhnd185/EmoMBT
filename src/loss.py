import torch
import torch.nn as nn
from end2you.utils import Params


class Criterion(nn.Module):
    def __init__(self, fusion, pos_weight) -> None:
        super().__init__()

        if fusion == 'dict':
            loss_dict = {
                "audio": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "text": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "visual": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            }
        else:
            loss_dict = {
                "audio": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "text": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "visual": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "fusion": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            }
        self.tasks = loss_dict.keys()
        self.losses = nn.ModuleDict(loss_dict)


    def forward(self, preds, targets):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task]

            task_losses.append(self.losses[task](pred, target))

        loss = torch.mean(torch.stack(task_losses))

        return loss


class UncertaintyCriterion(Criterion):
    """
    Criterion definition which adds restrained uncertainty to weigh the losses.
    """

    def __init__(self, fusion, pos_weight, device) -> None:
        super().__init__(fusion, pos_weight)
        # init parameters
        self.log_vars = nn.Parameter(torch.FloatTensor([1/len(self.tasks)] * len(self.tasks))).to(device)
        # constraint value
        self.phi = 1.0

    
    def forward(self, preds, targets):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task]

            task_losses.append(self.losses[task](pred, target))

        loss = torch.stack(task_losses)

        un_weights = 1 / (len(self.tasks) * self.log_vars ** 2)
        regularisation = torch.sum(torch.log(1 + self.log_vars ** 2))
        constraint = torch.abs(self.phi - torch.sum(torch.abs(self.log_vars)))

        loss = torch.sum(un_weights * loss) + regularisation + constraint

        return loss

class DynamicUncertaintyCriterion(Criterion):
    """
    Criterion definition which implements Dynamic Restrained Uncertainty Weighting for losses.
    See Song et al. (2022)
    """

    def __init__(self, fusion, pos_weight, device, temp) -> None:
        super().__init__(fusion, pos_weight)

        N = len(self.tasks)

        # init parameters
        self.log_vars = nn.Parameter(torch.ones(N) * 1 / N).to(device)
      
        self.phi = 1.0    # constraint value
        self.kappa = N   # scales the dynamic weights
        self.temperature = temp   # default value for smoothing softmax

        self.loss_t_1 = None
        self.loss_t_2 = None


    def forward(self, preds, targets):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task_index]

            task_losses.append(self.losses[task](pred, target))

            #task_loss = self.losses[task](pred, target)
            #if return_all:
            #    task_losses[task] = task_loss.item()    # update dict with value of single loss

        loss_t = torch.stack(task_losses)   # combine N 0-dimensional tensors into one tensor of size [N,]
        
        dyn_weights = dynamic_weight_average(num_tasks=len(self.tasks), kappa=self.kappa, temperature=self.temperature, loss_t1=self.loss_t_1, loss_t2=self.loss_t_2)
        dyn_weights = dyn_weights.to(loss_t.device) # move weights to the same device the loss tensor is on
        un_weights = 1 / (len(self.tasks) * self.log_vars ** 2)
        regularisation = torch.sum(torch.log(1 + self.log_vars ** 2))
        constraint = torch.abs(self.phi - torch.sum(torch.abs(self.log_vars)))
        
        loss = torch.sum((dyn_weights + un_weights) * loss_t) + regularisation + constraint

        # update states
        self.loss_t_2 = self.loss_t_1
        self.loss_t_1 = loss_t.detach()

        return loss


class DynamicWeightAverageCriterion(Criterion):
    def __init__(self, fusion, pos_weight, temp) -> None:
        super().__init__(fusion, pos_weight)

        self.kappa = len(self.tasks)
        self.temperature = temp

        self.loss_t_1 = None
        self.loss_t_2 = None


    def forward(self, preds, targets):
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):
            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task_index]

            task_losses.append(self.losses[task](pred, target))

        loss_t = torch.stack(task_losses)

        dyn_weights = dynamic_weight_average(num_tasks=len(self.tasks), kappa=self.kappa, temperature=self.temperature, loss_t1=self.loss_t_1, loss_t2=self.loss_t_2)
        dyn_weights = dyn_weights.to(loss_t.device)
        loss = torch.sum(dyn_weights * loss_t)

        # update states
        self.loss_t_2 = self.loss_t_1
        self.loss_t_1 = loss_t.detach()

        return loss




def criterion_factory(fusion, loss_strategy, temp, pos_weight, device) -> Criterion:
    """
    factory helper that generates the desired criterion based on the loss_strategy flag
    :params Params object, containing train.loss_strategy str, one of [mean, rruw, druw]
    """ 

    if loss_strategy == "mean":
        print("Averaging task losses")
        return Criterion(fusion, pos_weight)
    elif loss_strategy == "rruw":
        print("Using Restrained Revised Uncertainty Weighting for losses")
        return UncertaintyCriterion(fusion, pos_weight, device)
    elif loss_strategy == "druw":
        print("Using Dynamic Restrained Uncertainty Weighting for losses")
        return DynamicUncertaintyCriterion(fusion, pos_weight, device, temp)
    elif loss_strategy == "dwa":
        print("Using Dynamic Weight Averaging for losses")
        return DynamicWeightAverageCriterion(fusion, pos_weight, temp)
    else: 
        raise NotImplementedError("{} not implemented".format(loss_strategy))

    
def dynamic_weight_average(num_tasks:int, kappa:float, temperature:float, loss_t1, loss_t2) -> torch.FloatTensor:
        """
        computes dynamic loss weight from the losses of the last two steps 
        """

        if (loss_t1 is None) or (loss_t2 is None):  # if there are no previous time steps, return ones
            return torch.ones(num_tasks)
        
        assert len(loss_t1) == len(loss_t2), "Loss lists must have same number of tasks for each step"

        if isinstance(loss_t1, list):
            loss_t1 = torch.FloatTensor(loss_t1)
        if isinstance(loss_t2, list):
            loss_t2 = torch.FloatTensor(loss_t2)

        dl = loss_t1 / loss_t2
        dyn_weights = kappa * torch.softmax(dl / temperature, dim=0)

        assert isinstance(dyn_weights, torch.Tensor), "weights should be a Tensor but is {}".format(type(dyn_weights))

        return dyn_weights



