# Dataset creation

In order to create the dataset structures we used, you can execute the following code. 


### CropHarvest (binary and multi)  
```
python data_cropharvest.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR -c CROP_CASE
```
* options for CROP_CASE: [binary, multi]

> [!IMPORTANT]  
> The original data comes from [CropHarvest](https://github.com/nasaharvest/cropharvest)


### TreeSatAI-TS
```
python data_treesat.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR -s SPLIT
```
* options for SPLIT: [train, val, test]

> [!IMPORTANT]
> The original data comes from [TreeSatAI extension](https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series)


### Live Fuel Moisture Content  
```
python data_lfmc.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR 
```

> [!IMPORTANT]
> The original data comes from [LFMC from SAR](https://github.com/kkraoj/lfmc_from_sar)

 
---


> [!TIP] 
> Preprocessed data can be accessed at: [Link](https://cloud.dfki.de/owncloud/index.php/s/xJfLXQt22SffEk4)
