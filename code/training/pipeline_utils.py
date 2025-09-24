import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from code.datasets.utils import create_dataloader

def prepare_callback(data_name, method_name, run_id, fold_id, folder_c, tags_ml, monitor_name, **early_stop_args):
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    exp_folder_name = f'{data_name}/{method_name}'

    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}_{fold_id:02d}*'):
        v.unlink()
    
    callback_list = []
    if "mode" in early_stop_args:
        early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
        callback_list.append(early_stop_callback)
        mode_early = early_stop_args["mode"]
    else:
        mode_early = "min"
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=mode_early, every_n_epochs=1, save_top_k=1,
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}_{fold_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}')
    callback_list.append(checkpoint_callback)
    return {"callbacks":callback_list }

def build_dataloaders(train_data, val_data=None, batch_size=32, parallel_processes=2):
    if type(val_data) != type(None):
        val_dataloader = create_dataloader(val_data, batch_size=batch_size, train=False, parallel_processes=parallel_processes)
        monitor_name = "val_objective"
    else:
        val_dataloader = None
        monitor_name = "train_objective"
    train_dataloader = create_dataloader(train_data, batch_size=batch_size, parallel_processes=parallel_processes)
    return train_dataloader, val_dataloader, monitor_name


def get_shape_modality(modality_n, train_data, architecture={}):
    shape_modality_n = train_data.get_modality_shapes(modality_n)
    if len(architecture)!= 0 and modality_n in architecture["encoders"] != 0 and architecture["encoders"][modality_n]["model_type"] == "mlp":
        actual_shape_v = np.prod(shape_modality_n)
    elif len(architecture)!= 0 and modality_n in architecture["encoders"] != 0 and architecture["encoders"][modality_n]["model_type"] in ["predefined"]:
        actual_shape_v = shape_modality_n
    else:
        if len(shape_modality_n) < 3: #for time series data
            actual_shape_v = shape_modality_n[-1]
        else: #for other types of cubes it is flattened after time
            actual_shape_v = np.prod(shape_modality_n[1:])
    return actual_shape_v