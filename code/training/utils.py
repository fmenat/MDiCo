
import numpy as np
from sklearn.utils import class_weight

def assign_our_name(config_file, forward_modalities= [], perc:float=1, more_info_str=""):
    method_name = "Ours"
    
    if len(forward_modalities) != 0:
        method_name += "-Forw_" + "_".join(forward_modalities)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"

    return method_name + more_info_str


def assign_labels_weights(config_file, data_):
    
    if config_file.get("task_type", "").lower() == "classification":
        train_data_target = data_.get_all_labels().astype(int).flatten()
        config_file["training"]["loss_args"]["n_labels"] = train_data_target.max() +1
        config_file["training"]["loss_args"]["weight"] = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
    
    elif config_file.get("task_type", "").lower() == "multilabel":
        train_data_target = data_.get_all_labels()
        n_samples, n_labels = train_data_target.shape
        config_file["training"]["loss_args"]["n_labels"] = n_labels
        

def output_name(task_type):
    if task_type == "classification":
        return "softmax"
    elif task_type == "multilabel":
        return "sigmoid"
    elif task_type == "regression":
        return ""
    
