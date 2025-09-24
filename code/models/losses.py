import torch
from pytorch_metric_learning import losses


def get_loss_by_name(name, **loss_args):
    #https://pytorch.org/docs/stable/nn.html#loss-functions
    name = name.strip().lower().replace("_","")
    if "n_labels" in loss_args:
        loss_args = dict(loss_args)
        loss_args.pop("n_labels")
        
    if ("cross" in name and "entr" in name) or name=="ce":
        return torch.nn.CrossEntropyLoss(reduction="mean", **loss_args)
    elif ("bin" in name and "entr" in name) or name=="bce":
        return torch.nn.BCEWithLogitsLoss(reduction="mean", **loss_args)
    elif name == "kl" or name=="divergence": 
        return torch.nn.KLDivLoss(reduction="mean")
    elif name == "mse" or name =="l2":
        return torch.nn.MSELoss(reduction='mean')
    elif name == "mae" or name =="l1":
        return torch.nn.L1Loss(reduction='mean')
    elif name =="norm-mse":
        return normalize_mse(**loss_args)


def normalize_mse(mean, std, **kwargs):
    def real_loss(pred, real):
        return torch.nn.MSELoss(reduction='mean')((pred-mean)/std, (real-mean)/std)
    return real_loss


def pairwise_contrastive_loss(temperature=0.1, loss_type="infonce"):
    if loss_type == "infonce":
        loss_fn = losses.NTXentLoss(temperature=temperature)
    meta_loss = losses.SelfSupervisedLoss(loss_fn, symmetric=True)

    def real_loss(x_q, x_v):
        return meta_loss(embeddings=x_q, ref_emb=x_v)
    return real_loss
