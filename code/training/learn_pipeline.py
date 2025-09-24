from typing import List, Union, Dict
import copy
import numpy as np

import torch
import pytorch_lightning as pl

from code.models.mdico import MDiCo
from code.models.nn_module import create_model

from code.training.pipeline_utils import prepare_callback, build_dataloaders, get_shape_modality
        
def colearn_train(train_data: dict, val_data = None,
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="Ours", 
                     training = {}, architecture={}, **kwargs):
    folder_c = output_dir_folder+"/run-saves"
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]

    if "weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    elif "pos_weight" in loss_args:
        n_labels = loss_args.pop("n_labels")
        loss_args["pos_weight"] = torch.tensor(loss_args["pos_weight"],dtype=torch.float)
    else:
        n_labels = loss_args.get("n_labels", 1)

    args_model =  {"loss_args":loss_args, **training.get("additional_args", {})}
           
    #MODEL DEFINITION -- ENCODER
    pred_base = create_model(emb_dim//2, None, just_base=True, **architecture["predictive_model"]) 
    
    modalities_encoder  = {}
    for modality_n in architecture["encoders"]:
        actual_shape_v = get_shape_modality(modality_n, train_data, architecture=architecture)
        modalities_encoder[modality_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"][modality_n])
        
    model = MDiCo(modalities_encoder, pred_base, n_labels=n_labels, **args_model)
    print("Initial parameters of model:", model.hparams_initial)
    
    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"]) #,profiler="simple")
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    if training.get("inference"):
        model.set_aggregation(training.get("inference")["mode"])

    return model, trainer