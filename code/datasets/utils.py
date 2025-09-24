import os
import torch
import xarray as xray
import numpy as np
from typing import List, Union, Dict

from .multimodal import Dataset_MultiModal


def create_dataloader(dataset_pytorch, batch_size=32, train=True, parallel_processes=2, **args_loader):
    cpu_count = len(os.sched_getaffinity(0))
    return torch.utils.data.DataLoader(
        dataset_pytorch,
        batch_size=batch_size,
        num_workers=int(cpu_count/parallel_processes),
        shuffle=train,
        pin_memory=True,
        drop_last=False, 
        **args_loader
    )


def xray_to_datamodalities(xray_data: xray.Dataset, modalities_used: List[str]=[]) -> Dataset_MultiModal:
    all_possible_index = xray_data.coords["identifier"].values
    
    datamodalities = Dataset_MultiModal()    
    datamodalities.train_identifiers = list(all_possible_index[xray_data["train_mask"].values])
    datamodalities.val_identifiers = list(all_possible_index[~xray_data["train_mask"].values])
    datamodalities.target_names = xray_data.attrs["target_names"]
    datamodalities.modality_names = xray_data.attrs["view_names"]

    datamodalities.identifiers_target = dict(zip(all_possible_index, xray_data["target"]))
    for modality_n in (datamodalities.modality_names if len(modalities_used) == 0 else modalities_used):
        datamodalities.modalities_data_ident2indx[modality_n] = dict(zip(all_possible_index, np.arange(len(all_possible_index))))
        datamodalities.modalities_data[modality_n] = xray_data[modality_n] 
    
    return datamodalities

def load_structure(path: str, file_name: str, load_memory: bool = False, modalities_used: List[str]=[]) -> Dataset_MultiModal:
    data  = xray.open_dataset(f"{path}/{file_name}.nc", engine="h5netcdf")
    if load_memory:
        data = data.load()
    dataset_structure =  xray_to_datamodalities(data, modalities_used=modalities_used)
    data.close()
    return dataset_structure
