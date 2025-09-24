import numpy as np
import torch
from typing import List, Union, Dict


def detach_all(z):
    if isinstance(z, dict):
        z_ = {}
        for k, v in z.items():
            z_[k] = detach_all(v)
        z = z_
    elif isinstance(z, list):
        z = [z_.detach().cpu().numpy() for z_ in z]
    else:
        z = z.detach().cpu().numpy()
    return z

def collate_all_list(z, z_):
    if isinstance(z_, dict):
        for k, v in z_.items():
            collate_all_list(z[k], v)
    elif isinstance(z_,list):
        for i, z_i in enumerate(z_):
            z[i].append( z_i )
    else:
        z.append(z_) 

def object_to_list(z):
    if isinstance(z, dict):
        for k, v in z.items():
            z[k] = object_to_list(v)
        return z
    elif isinstance(z,list):
        return [ [z_i] for z_i in z]
    else:
        return [z]

def stack_all(z_list, data_type = "numpy"):
    if isinstance(z_list, dict):
        for k, v in z_list.items():
            z_list[k] = stack_all(v, data_type=data_type)
    elif isinstance(z_list[0], list):
        for i, v in enumerate(z_list):
            z_list[i] = stack_all(v, data_type=data_type)
    elif isinstance(z_list, list):
        if data_type == "numpy":
            z_list = np.concatenate(z_list, axis = 0)
        elif data_type == "torch":
            z_list = torch.concat(z_list, axis = 0)
    else:
        print(type(z_list))
        pass
    return z_list

def count_parameters(model) -> dict:
    total_trainable_params = 0
    total_non_trainable_params = 0
    save_array = {}
    for name, module in model.named_children():
        param = sum(p.numel() for p in module.parameters() if p.requires_grad) #parameter.numel()
        if len(list(module.named_children())) > 1 and param != 0:
            vals = count_parameters(module) #recursevily count all inside
        else:
            vals = {}
        save_array[name] = param
        save_array[name+"_ext"] = vals
        total_trainable_params+=param
        total_non_trainable_params+= sum(p.numel() for p in module.parameters() if not p.requires_grad)
    save_array["Total trainable param"] = total_trainable_params
    save_array["Total non-trainable param"] = total_non_trainable_params
    save_array["Total params"] = total_trainable_params
    return save_array


#FROM: https://github.com/Mikoto10032/AutomaticWeightedLoss
class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
        in regular model learn param = log(param**2)
            in corrected model learn param**2 directly
    """
    def __init__(self, num: int, type: str, loss_names=[]):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params) 
        self.type = type
        self.losses_names = loss_names

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            if self.losses_names[i] == None:
                continue

            if self.type =="uncertainty":
                loss_sum += torch.exp(-self.params[i]) * loss + self.params[i]
            elif self.type == "corrected":
                loss_sum += (0.5 / (self.params[i]) )* loss + torch.log(1 + self.params[i])
            else:
                raise Exception("type must be 'uncertainty' or 'corrected'")
        return loss_sum
    
    def get_params(self):
        print(self.params)
        if self.type in ["uncertainty"]:
            return dict(zip(self.losses_names,  torch.exp(-self.params).detach().cpu().numpy()))
        elif self.type == "corrected":
            return dict(zip(self.losses_names,  (0.5 / self.params).detach().cpu().numpy()))
