import numpy as np
import pandas as pd
import xarray as xray
from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm

from torch.utils.data import Dataset

class Dataset_MultiModal(Dataset):
    """
    Example: one item, one data example, could contain several modalities. However, fullmodal scenario is considered.
    n-modalities: number of modalities
    n-examples: number of examples
    
    Attributes
    ----------
        modalities_data : dictionary 
            with the data {key:name}: {modality name:array of data in that modality}
        modalities_data_ident2indx : dictionary of dictionaries
            with the data {modality name: dictionary {index:identifier} }
        inverted_ident : dictionary 
            with {key:indx}: {identifier:list of modalities name (index) that contain that identifier}
        modality_names : list of string
            a list with the modality names
        modalities_cardinality : dictionary 
            with {key:name}: {modality name: n-examples that contain that modality}
        identifiers_target: dictionary
            with the target corresponding ot each index example
        target_names : list of strings
            list with string of target names, indicated in the order
    """ 
    def __init__(self, modalities_to_add: Union[list, dict] = [], identifiers:List[int] = [], modality_names: List[str] =[] , target: List[int] =[]):

        """initialization of attributes. You also could given the modalities to add in the init function to create the structure already, without using add_modality method

        Parameters
        ----------
            modalities_to_add : list or dict of numpy array, torch,tensor or any
                the modalities to add
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the modalities_to_add
            modality_names : list of string 
                the name of the modalities being added
            target: list of int
                the target values if available (e.g. supervised data)
        """
        self.modalities_data = {}
        self.modalities_data_ident2indx = {} 
        self.modality_names = []
        self.identifiers_target = {}
        self.target_names = []
        self.train_identifiers = []
        self.val_identifiers = []
        self.train_set = True

        self.stats_xarray = None
        self.set_additional_info()
        
        if len(modalities_to_add) != 0:
            if len(identifiers) == 0:
                identifiers = np.arange(len(modalities_to_add[0]))
            if len(modality_names) == 0 and type(modalities_to_add) != dict:
                modality_names = ["S"+str(v) for v in np.arange(len(modalities_to_add))]
            elif len(modality_names) == 0 and type(modalities_to_add) == dict:
                modality_names = list(modalities_to_add.keys())

            for v in range(len(modalities_to_add)):
                if type(modalities_to_add) == list or type(modalities_to_add) == np.ndarray:
                    self.add_modality(modalities_to_add[v], identifiers, modality_names[v])
                if type(modalities_to_add) == dict:
                    self.add_modality(modalities_to_add[modality_names[v]], identifiers, modality_names[v])
            if len(target) != 0:
                self.add_target(target, identifiers)
            else:
                self.train_identifiers = identifiers
        
    def add_target(self, target_to_add: Union[list,np.ndarray], identifiers: List[int], target_names: List[str] = []):
        """add a target for the corresponding identifiers indicated, it also works by updating target

        Parameters
        ----------
            target_to_add : list, np.array or any structure that could be indexed as ARRAY[i]
                the target values to add 
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the target_to_add
            target_names : list of str 
                the target names if available (e.g. categorical targets)
        """
        for i, ident in enumerate(identifiers): 
            self.identifiers_target[ident] = target_to_add[i]
        self.target_names = [f"T{i}" for i in range(len(self.identifiers_target[identifiers[-1]]))] if len(target_names) == 0 else target_names
        self.train_identifiers = list(self.identifiers_target.keys())

    def add_modality(self, modality_to_add, identifiers: List[int], name: str):
        """add a modality array based on identifiers of list and name of the modality. The identifiers is used to match the modality with others modalities.

        Parameters
        ----------
            modality_to_add : ideally xarray.DataArray, but numpy array is also possible 
                the array of the modality to add (no restriction in dimensions or shape)
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the modality_to_add
            name : string 
                the name of the modality being added
        """
        if name in self.modality_names:
            print("This modality is already saved, it will be updated")
        self.modality_names.append(name)
        self.modalities_data[name] = modality_to_add
        self.modalities_data_ident2indx[name] = {}
        for indx, ident in enumerate(identifiers):
            self.modalities_data_ident2indx[name][ident] = indx

    def set_val_mask(self, identifiers: List[int]):
        """set a binary mask to indicate the test examples

        Parameters
        ----------
            identifiers : list of identifiers that correspond to the test examples

        """
        self.train_identifiers = list( set(self.identifiers_target.keys()) - set(identifiers) )
        self.val_identifiers = identifiers

    def set_data_mode(self, train: bool):
        self.train_set = train

    def set_used_modality_names(self, modality_names: List[str]):
        self.used_modality_names = list(modality_names)
        self.extended_used_modality_names = list(self.used_modality_names)
        for v in modality_names:
            if "_" in v:
                self.extended_used_modality_names.extend(v.split("_"))

    def set_additional_info(self,form="zscore",fillnan=False,fillnan_value=0.0,flatten=False,modality_names=[]):
        self.form = form
        self.fillnan = fillnan
        self.fillnan_value = fillnan_value
        self.flatten = flatten

        if len(modality_names) != 0:
            self.set_used_modality_names(modality_names)
        else:
            self.used_modality_names = modality_names
            self.extended_used_modality_names = modality_names
        

    def __len__(self) -> int:
        return len(self.get_all_identifiers())

    def get_all_identifiers(self) -> list:
        """get the identifiers of all modalities on the corresponding set
     
        Returns
        -------
            list of identifiers
            
        """
        if len(self.val_identifiers) != 0:
            identifiers = self.train_identifiers if self.train_set else self.val_identifiers
        else:
            identifiers = self.train_identifiers
        return identifiers
    
    def get_all_labels(self):
        identifier = self.get_all_identifiers()
        labels = []
        for ident in identifier:
            labels.append(self.identifiers_target[ident])
        return np.array(labels)
    
    def get_modality_data(self, name: str):
        identifier = self.get_all_identifiers()

        data_modalities = []
        for ident in identifier:
            data_modalities.append(self.modalities_data[name][self.modalities_data_ident2indx[name][ident]])
        return {"modalities": np.stack(data_modalities, axis=0), "identifiers":identifier  , "modality_names": [name]}
    
    
    def normalize_w_stats(self, data, modality_name):
        mean_ = self.stats_xarray[f"{modality_name}-mean"].values
        std_ = self.stats_xarray[f"{modality_name}-std"].values
        max_ = self.stats_xarray[f"{modality_name}-max"].values
        min_ = self.stats_xarray[f"{modality_name}-min"].values

        if self.form == "zscore":
            return (data - mean_)/std_
        elif self.form == 'max': 
            return data/max_
        elif self.form == "minmax-01":
            return (data - min_)/(max_ - min_)


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, List[np.ndarray], List[str]]]:
        """
        Parameters
        ----------
            index : int value that correspond to the example to get (with all the modalities available)

        Returns
        -------
            dictionary with three values
                data : numpy array of the example indicated on 'index' arg
                modalities : a list of strings with the modalities available for that example
                train? : a mask indicated if the example is used for train or not    
        """
        identifier = self.get_all_identifiers()[index]

        target_data = self.identifiers_target[identifier] 
        target_data = target_data.values if type(target_data) == xray.DataArray else target_data
        
        modality_data = {}
        modalities_to_add = []
        for i, modality in enumerate((self.modality_names if len(self.extended_used_modality_names) == 0 else self.extended_used_modality_names)):
            if modality not in self.modality_names:
                if "_" in modality:
                    modalities_to_add.append(modality)
                continue

            data_ = self.modalities_data[modality]
                
            if type(data_) == np.ndarray:
                data_ = data_[self.modalities_data_ident2indx[modality][identifier]]
            if type(data_) == xray.DataArray:
                data_ = data_.isel(identifier=self.modalities_data_ident2indx[modality][identifier]).values
            else:
                raise Exception(f"Data type for modality {modality} not supported")
            
            if self.stats_xarray is not None:
                data_ = self.normalize_w_stats(data_, modality)
            if self.fillnan:
                data_ = np.nan_to_num(data_, nan=self.fillnan_value)
            if self.flatten:
                data_ = data_.reshape(-1)

            modality_data[modality] = data_.astype("float32")
            
        for modality in modalities_to_add:
            data_list = []
            for v in modality.split("_"):
                data_list.append(modality_data.pop(v))
            modality_data[modality] = np.concatenate(data_list, axis=-1)

        return {"identifier": identifier,
                "modalities": modality_data, 
                "target": target_data,
                }
        
    def get_modality_names(self, indexs: List[int] = []) -> List[str]:
        """get the names of the modalities available in the corresponding indices"""
        return self.modality_names if len(indexs) == 0 else np.asarray(self.modality_names)[indexs].tolist()

    def get_modality_shapes(self, modality_name:str ="") -> Dict[str, tuple]:
        return_dic = {name: self.modalities_data[name].shape[1:] for name in self.modality_names}
        if modality_name in return_dic:
            return return_dic[modality_name]
        elif "_" in modality_name:
            return (return_dic[modality_name.split("_")[0]][:-1]) +(sum([return_dic[v][-1] for v in modality_name.split("_")]),)

    def _to_xray(self):
        data_vars = {}
        for modality_n in self.get_modality_names():
            data_vars[modality_n] = xray.DataArray(data=self.modalities_data[modality_n], 
                                  dims=["identifier"] +[f"{modality_n}-D{str(i+1)}" for i in range (len(self.modalities_data[modality_n].shape)-1)], 
                                 coords={"identifier":list(self.modalities_data_ident2indx[modality_n].keys()),
                                        }, )
        
        data_vars["train_mask"] = xray.DataArray(data=np.asarray([1]*len(self.train_identifiers) + [0]*len(self.val_identifiers), dtype="bool"),  
                                        dims=["identifier"], 
                                         coords={"identifier": self.train_identifiers + self.val_identifiers })
        if len(self.identifiers_target) != 0:
            data_vars["target"] = xray.DataArray(data = np.stack(list(self.identifiers_target.values()), axis=0),
                dims=["identifier","dim_target"] , coords ={"identifier": list(self.identifiers_target.keys())} ) 

        return xray.Dataset(data_vars =  data_vars,
                        attrs = {
                                "modality_names": self.modality_names, 
                                 "target_names": self.target_names,
                                },
                        )
        
    def save(self, name_path, xarray = True, ind_modalities = False):
        """save data in name_path

        Parameters
        ----------
            name_path : path to a file to save the model, without extension, since extension is '.pkl.
            ind_modalities : if you want to save the individual modalities as csv files 
        """
        path_ = Path(name_path)
        name_path_, _, file_name_ = name_path.rpartition("/") 
        path_ = Path(name_path_)
        path_.mkdir(parents=True, exist_ok=True)

        if xarray and (not ind_modalities): 
            xarray_data = self._to_xray()
            path_ = path_ / (file_name_+".nc" if "nc" != file_name_.split(".")[-1] else file_name_)
            xarray_data.to_netcdf(path_, engine="h5netcdf") 
            return xarray_data
        elif (not xarray) and ind_modalities:  #only work with 2D arrays
            path_ = Path(name_path_ +"/"+ file_name_)
            path_.mkdir(parents=True, exist_ok=True)
            for modality_name in self.get_modality_names():
                modality_data_aux = self.get_modality_data(modality_name)
                df_tosave = pd.DataFrame(modality_data_aux["modalities"])
                df_tosave.index = modality_data_aux["identifiers"]
                df_tosave.to_csv(f"{str(path_)}/{modality_name}.csv", index=True)

    def load_stats(self, path:str, file_name:str ):
        self.stats_xarray = xray.open_dataset(f"{path}/stats/stats_{file_name}.nc", engine="h5netcdf").load()