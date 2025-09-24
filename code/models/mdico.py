import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

from .core_model import _BaseModalitiesLightning
from .losses import get_loss_by_name, pairwise_contrastive_loss
from .single.base_decoders import Generic_Decoder
from .utils import stack_all, object_to_list, collate_all_list, detach_all, AutomaticWeightedLoss

class MDiCo(_BaseModalitiesLightning):
    def __init__(self,
                 modality_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 predictive_core: nn.Module,
                 n_labels: int,
                 loss_args: dict ={},
                 weights_loss: dict = {},
                 contrastive_temp: float = 0.1,
                 **kwargs,
                 ):
        super(MDiCo, self).__init__(**kwargs)          
        self.modality_names = list(modality_encoders.keys())
        self.N_modalities = len(self.modality_names)
        self.n_labels = n_labels
        self.weights_loss = weights_loss
        self.contrastive_temp = contrastive_temp

        #for encoding unique and common information
        self.modality_encoders_unique = {}
        self.modality_encoders_common = {}
        self.aux_common_layer = {}
        self.modality_prediction_heads = {}
        for v_name ,model_v in modality_encoders.items():
            self.modality_encoders_unique[v_name] = copy.deepcopy(model_v)
            self.modality_encoders_common[v_name] = copy.deepcopy(model_v)
            emb_dim = model_v.get_output_size() //2
            self.aux_common_layer[v_name] =  nn.Sequential(nn.Linear(emb_dim*2, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())

            #for main prediction
            self.modality_prediction_heads[v_name] = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=self.n_labels, input_dim=emb_dim*2)
            self.modality_prediction_heads[v_name].update_first_layer(emb_dim*2)
        self.modality_encoders_unique = nn.ModuleDict(self.modality_encoders_unique)
        self.modality_encoders_common = nn.ModuleDict(self.modality_encoders_common)
        self.aux_common_layer = nn.ModuleDict(self.aux_common_layer)
        self.modality_prediction_heads = nn.ModuleDict(self.modality_prediction_heads)

        #for auxiliary prediction
        self.aux_prediction_head = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=self.n_labels, input_dim=emb_dim)

        #for modality prediction
        self.modality_head_adv = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=len(self.modality_names), input_dim=emb_dim)
        self.modality_head_awr_inf = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=len(self.modality_names), input_dim=emb_dim)
        self.modality_head_awr_irr = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=len(self.modality_names), input_dim=emb_dim)
        
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.criteria_modality = nn.CrossEntropyLoss()
        self.criteria_contrastive = pairwise_contrastive_loss(self.contrastive_temp, loss_type="infonce")
        
        self.save_hyperparameters(ignore=["modality_encoders","predictive_core"])

    def prepare_batch(self, batch: dict) -> list:
        modalities_dict, modalities_target = batch["modalities"], batch["target"]
        
        if type(self.criteria) == torch.nn.CrossEntropyLoss:
            modalities_target = modalities_target.squeeze().to(torch.long)
        else:
            modalities_target = modalities_target.to(torch.float32)

        return modalities_dict, modalities_target
    
    def forward_encoders(self, modalities: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
        zs_modalities = {}
        for v_name in self.modality_names:
            zs_modalities[v_name] = self.modality_encoders[v_name](modalities[v_name])
        return zs_modalities

    def forward(self, modalities: Dict[str, torch.Tensor], out_norm:str = "", intermediate=True, **kwargs):
        self.modality_encoders = self.modality_encoders_unique
        out_zs_modalities_spe = self.forward_encoders(modalities)
        
        self.modality_encoders = self.modality_encoders_common
        out_zs_modalities_sha = self.forward_encoders(modalities)
                
        main_out_pred = {}
        aux_out_y_zs = {"specific": {}, "shared": {}}
        aux_out_mod_discr = {"specific": {}, "unused": {}}
        aux_reps = {"specific":{}, "shared":{}, "unused":{}}
        for v_name in out_zs_modalities_spe.keys():
            spec_full = out_zs_modalities_spe[v_name]
            sha_full = out_zs_modalities_sha[v_name]
            
            nfeat = spec_full.shape[1]//2
            spe_inf, spe_irr = spec_full[:, :nfeat], spec_full[:, nfeat:]
            sha_inv = self.aux_common_layer[v_name](sha_full)
            aux_reps["specific"][v_name] = spe_inf
            aux_reps["unused"][v_name]  = spe_irr
            aux_reps["shared"][v_name]   = sha_inv
            
            #main prediction for L_MAIN
            out_y = self.modality_prediction_heads[v_name](torch.cat([aux_reps["specific"][v_name], aux_reps["shared"][v_name]],axis=-1)) 
            main_out_pred[v_name] = self.apply_norm_out(out_y, out_norm) #normalize output (softmax/sigmoid) when is used in inference

            #auxiliary prediction for L_AUX
            aux_out_y_zs["specific"][v_name] = self.aux_prediction_head(aux_reps["specific"][v_name]) 
            aux_out_y_zs["shared"][v_name] = self.aux_prediction_head(aux_reps["shared"][v_name]) 

            #auxiliary prediction for L_MOD
            aux_out_mod_discr["specific"][v_name] = self.modality_head_awr_inf(aux_reps["specific"][v_name]) 
            aux_out_mod_discr["unused"][v_name] = self.modality_head_awr_irr(aux_reps["shared"][v_name] ) 
            
        return_dict = {"modalities:prediction": main_out_pred, 
                       "modalities:aux:prediction":aux_out_y_zs}
        if intermediate:
            return_dict["modalities:rep"] = aux_reps
            return_dict["modalities:aux:modality_discr"] = aux_out_mod_discr
        return return_dict    
    
    def loss_batch(self, batch: dict):
        modalities_dict, target_all = self.prepare_batch(batch)
        
        out_dic = self(modalities_dict)

        pred_modality = out_dic["modalities:prediction"]
        aux_pred_modality = out_dic["modalities:aux:prediction"]
        aux_discr_modality = out_dic["modalities:aux:modality_discr"]
        rep_modality = out_dic["modalities:rep"]
        modality_names = list(pred_modality.keys())

        loss_dic = {}
        loss_main = 0
        loss_aux = 0
        loss_contr = 0
        loss_mod = 0
        for v_i, v_name in enumerate(modality_names):
            target_modality = target_all

            #Main predictive losses
            loss_dic["Lmain"+v_name] = self.criteria(pred_modality[v_name], target_modality)
            loss_main+= self.weights_loss.get("main",1)*loss_dic["Lmain"+v_name]/len(modality_names)

            #Auxiliary predictive losses
            if self.weights_loss.get("auxiliary",0) != 0: #loss is 'activated'
                loss_dic["Laux"+v_name] = (self.criteria(aux_pred_modality["specific"][v_name], target_modality) + self.criteria(aux_pred_modality["shared"][v_name], target_modality))/2
                loss_aux += self.weights_loss["auxiliary"]*loss_dic["Laux"+v_name]/len(modality_names) 

            #Contrastive losses
            if self.weights_loss.get("contrastive",0) != 0:
                #cross-modality/sensor contrastive loss
                for v_j in range(len(modality_names)): 
                    if v_i == v_j: #do not compute loss with itself
                        continue
                    loss_dic["Lcon"+modality_names[v_i]+"-"+modality_names[v_j]] = self.criteria_contrastive(rep_modality["shared"][modality_names[v_i]], rep_modality["shared"][modality_names[v_j]])
                    loss_contr += self.weights_loss["contrastive"]*loss_dic["Lcon"+modality_names[v_i]+"-"+modality_names[v_j]] 

            #Modality discriminative losses
            if self.weights_loss.get("discriminative", 0) != 0: #loss is 'activated'
                label_ = np.where(v_name == np.asarray(self.modality_names))[0].item()
                modality_labels = torch.ones_like(aux_discr_modality["specific"][v_name][:,0], dtype=torch.long)*label_
                loss_dic["Ldisc"+v_name] = (self.criteria_modality(aux_discr_modality["specific"][v_name], modality_labels) + self.criteria_modality(aux_discr_modality["unused"][v_name], modality_labels) )/2
                loss_mod += self.weights_loss["discriminative"]*loss_dic["Ldisc"+v_name]/len(modality_names) 

        total_loss = loss_main + loss_aux +  loss_contr + loss_mod 
    
        return {"objective": total_loss, **loss_dic}
    
    def get_encoder_models(self):
        return self.modality_encoders
    
    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate: bool=True,
            out_norm:str="",
            device:str="",
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed modalities

        #return numpy arrays based on dictionary
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                modalities_dict, _ = self.prepare_batch(batch)
                for modality_name in modalities_dict:
                    modalities_dict[modality_name] = modalities_dict[modality_name].to(device_used)

                outputs_ = self(modalities_dict, intermediate=intermediate, out_norm=out_norm)
                
                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu
    