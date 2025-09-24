import yaml
import argparse
import os
import sys
import time
import gc
import copy
from pathlib import Path
import numpy as np
import pandas as pd

from code.training.learn_pipeline import colearn_train
from code.training.utils import assign_our_name, output_name, assign_labels_weights
from code.datasets.multimodal import Dataset_MultiModal
from code.datasets.utils import create_dataloader, load_structure

def main_run(config_file):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    runs_seed = config_file["experiment"].get("runs_seed", [])
    if len(runs_seed) == 0:
        runs = config_file["experiment"].get("runs", 1)
        runs_seed = [np.random.randint(50000) for _ in range(runs)]

    BS = config_file["training"]["batch_size"]
    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    if config_file.get("task_type", "").lower() == "classification":
        config_file["training"]["loss_args"]["name"] = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "regression":
        config_file["training"]["loss_args"]["name"] = "mse" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "multilabel":
        config_file["training"]["loss_args"]["name"] = "bce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    method_name = assign_our_name(config_file, more_info_str=config_file.get("additional_method_name", ""))

    if "train" in data_name:
        data_modalities_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_modalities_tr.load_stats(input_dir_folder, data_name)
        data_modalities_te = load_structure(input_dir_folder, data_name.replace("train", "test"), load_memory=config_file.get("load_memory", False))
        data_modalities_te.load_stats(input_dir_folder, data_name)
        try:
            data_modalities_va = load_structure(input_dir_folder, data_name.replace("train", "val"), load_memory=config_file.get("load_memory", False))
            data_modalities_va.load_stats(input_dir_folder, data_name)
        except:
            data_modalities_va = data_modalities_te
            print("No validation set found")
        kfolds = 1
    else:
        data_modalities_tr = load_structure(input_dir_folder, data_name, load_memory=config_file.get("load_memory", False))
        data_modalities_tr.load_stats(input_dir_folder, data_name)
        kfolds = config_file["experiment"].get("kfolds", 2)

    for r,r_seed in enumerate(runs_seed):
        np.random.seed(r_seed)
        if kfolds != 1:
            indexs_ = data_modalities_tr.get_all_identifiers() 
            np.random.shuffle(indexs_)
            indexs_runs = np.array_split(indexs_, kfolds)
        for k in range(kfolds):
            print(f"******************************** Executing model on run {r+1} and kfold {k+1}")
            
            if kfolds != 1:
                data_modalities_tr.set_val_mask(indexs_runs[k])
                data_modalities_te = copy.deepcopy(data_modalities_tr)
                data_modalities_te.set_data_mode(train=False)
                data_modalities_va = data_modalities_te
            data_modalities_tr.set_additional_info(modality_names=list(config_file["architecture"]["encoders"].keys()),**config_file["experiment"].get("preprocess"))
            data_modalities_va.set_additional_info(modality_names=list(config_file["architecture"]["encoders"].keys()),**config_file["experiment"].get("preprocess"))
            data_modalities_te.set_additional_info(modality_names=list(config_file["architecture"]["encoders"].keys()),**config_file["experiment"].get("preprocess"))
            print(f"Training with {len(data_modalities_tr)} samples and validating on {len(data_modalities_te)}")

            if config_file.get("task_type", "").lower() in ["classification", "multilabel"]:
                assign_labels_weights(config_file, data_modalities_tr)
                
            method, trainer = colearn_train(data_modalities_tr, val_data=data_modalities_va,run_id=r,fold_id=k, method_name=method_name,  **config_file)
            print("Training done")

            outputs_te = method.transform(create_dataloader(data_modalities_te, batch_size=BS, train=False), out_norm=output_name(config_file["task_type"]), intermediate=config_file["training"].get("return_rep"))
            
            #main per-modality predictions
            for modality_n, values in outputs_te["modalities:prediction"].items():
                aux_name = assign_our_name(config_file, more_info_str=f"_{modality_n}"+config_file.get("additional_method_name", ""))
                data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_modalities=True, xarray=False)
            

            #additional auxiliar outputs
            if "prediction" in outputs_te:
                data_save_te = Dataset_MultiModal([outputs_te["prediction"]], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                data_save_te.save(f"{output_dir_folder}/pred/{data_name}/"+method_name.replace("Ours", "EnsembleOurs"), ind_modalities=True, xarray=False)
                
            if "modalities:aux:prediction" in outputs_te: 
                for modality_n, values in outputs_te["modalities:aux:prediction"]["specific"].items():
                    aux_name = assign_our_name(config_file, more_info_str=f"_Spe{modality_n}"+config_file.get("additional_method_name", ""))
                    data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_modalities=True, xarray=False)                
                for modality_n, values in outputs_te["modalities:aux:prediction"]["shared"].items():
                    aux_name = assign_our_name(config_file, more_info_str=f"_Sha{modality_n}"+config_file.get("additional_method_name", ""))
                    data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_modalities=True, xarray=False)    
            
            if "modalities:rep" in outputs_te:
                for modality_n, values in outputs_te["modalities:rep"]["specific"].items():
                    aux_name = assign_our_name(config_file, more_info_str=f"Spe_{modality_n}"+config_file.get("additional_method_name", ""))
                    data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/repre/{data_name}/{aux_name}", ind_modalities=True, xarray=False)
                
                for modality_n, values in outputs_te["modalities:rep"]["shared"].items():
                    aux_name = assign_our_name(config_file, more_info_str=f"Sha_{modality_n}"+config_file.get("additional_method_name", ""))
                    data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/repre/{data_name}/{aux_name}", ind_modalities=True, xarray=False)

                for modality_n, values in outputs_te["modalities:rep"]["unused"].items():
                    aux_name = assign_our_name(config_file, more_info_str=f"Irr_{modality_n}"+config_file.get("additional_method_name", ""))
                    data_save_te = Dataset_MultiModal([values], identifiers=data_modalities_te.get_all_identifiers(), modality_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/repre/{data_name}/{aux_name}", ind_modalities=True, xarray=False)


            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {method_name} finished...")

    print(f"Finished whole execution of {len(runs_seed)} runs in {time.time()-start_time:.2f} secs")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file)
