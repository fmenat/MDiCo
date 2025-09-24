from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, balanced_accuracy_score,matthews_corrcoef
from sklearn.metrics import r2_score, mean_tweedie_deviance

import numpy as np

# ALL CLASSIFICATION METRICS HAS (y_pre, y_true) order!!
class OverallAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(y_pred == y_true)
	
class AverageAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return balanced_accuracy_score(y_true, y_pred)
	
class F1Score():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		return f1_score(y_true, y_pred, average=self.average)

class Precision():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return precision_score(y_true, y_pred, average=self.average)
    
class Recall():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return recall_score(y_true, y_pred, average=self.average)
    
class Kappa():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return cohen_kappa_score(y_true, y_pred)
	
class MCC():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return matthews_corrcoef(y_true, y_pred)
	
class ConfusionMatrix():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return confusion_matrix(y_true, y_pred)
	
class Get_Data():
	def __init__(self, task_type="classification", which="true"):
		self.which = which
		self.task_type = task_type

	def __call__(self, y_pred,y_true):
		data_check = y_true if self.which == "true" else y_pred
		if self.task_type == "classification":
			labels_n = np.unique(y_true)
			return [np.sum(data_check==v) for v in labels_n]
		elif self.task_type == "multilabel":
			labels_n = np.arange(y_true.shape[-1])
			return [ data_check[:,v].sum(axis=0) for v in labels_n]
	


class R2Score():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return r2_score(y_true, y_pred)
	
class MAE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(np.abs(y_true -y_pred))
	
class MedAE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.median(np.abs(y_true -y_pred))
	
class RMSE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.sqrt(np.mean((y_true - y_pred)**2))
	
class rRMSE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.sqrt(np.mean((y_true - y_pred)**2/np.mean(y_true)))
	
class MAPE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(np.abs((y_true - y_pred)/y_true))
	
class BIAS():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(y_pred - y_true)
	
class TweediePoisson():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return mean_tweedie_deviance(y_true, y_pred, power=1)
	
class PCorr():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		y_pred_mean = y_pred.mean()
		y_pred_std = y_pred.std()
		y_true_mean = y_true.mean()
		y_true_std = y_true.std()

		covariance_ = np.mean((y_pred-y_pred_mean)*(y_true-y_true_mean))

		return covariance_/(y_pred_std*y_true_std)