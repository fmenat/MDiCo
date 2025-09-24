# MDiCo: Multi-modal Disentanglement for Co-learning with Earth Observation data

Public repository of our work in multi-modal co-learning for missing modality with EO data.

![missing data](imgs/mdico_model.jpg)

The previous image illustrates our **MDiCo** framework in a multi-modal setting. We focus on the co-learning with multi-sensor Earth observation data, including classification (binary, multi-class, multi-label) and regression tasks. The objective is to achieve robust models for the all-but-one missing modality scenarios, i.e. multi-modal data available for training and a single-modality data available for inference.


## Training
We provide config file examples on how to train our model with different settings.

* To train a method based on MDiCo framework in the CropHarvest multi dataset, run:  
```
python train.py -s config/cropmulti_full.yaml
```
For other datasets you can check the other config files in the  [config folder](./config).

> [!NOTE]  
> Read about the used data in [data folder](./data)


## Evaluation
![missing data](imgs/missing_data.jpg)

* To evaluate the predictive performance run:
```
python evaluate.py -s config/eval.yaml
```
All details to folder paths and configurations are inside the yaml files.

---


# :scroll: Source

* nothing yet


# 🖊️ Citation

...
