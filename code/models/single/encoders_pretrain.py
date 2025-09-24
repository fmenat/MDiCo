import torch
import torch.nn as nn
import torchvision.models as models

from .sota_aux.conv_blocks import Conv2DBlock

class ImageNetExtractor(nn.Module):
    """
    Initialize a ResNet for feature extraction
    """
    def __init__(self, n_bands, model_name="resnet50", **kwargs):
        super(ImageNetExtractor, self).__init__()
        self.model_name = model_name
        
        if self.model_name == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        elif self.model_name == "resnet34":
            self.model = models.resnet34(weights="IMAGENET1K_V1")
        elif self.model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V2")
        elif self.model_name == "resnet101":
            self.model = models.resnet101(weights="IMAGENET1K_V2")
        elif self.model_name == "resnet152":
            self.model = models.resnet152(weights="IMAGENET1K_V2")

        if n_bands != 3:
            self.model.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #pop last layer
        self.out_size =  self.model.fc.in_features
        self.model.fc = nn.Identity()
        
    def forward(self, x, **kwargs):
        if x.shape[-1] != x.shape[-2]:
            x = x.permute(0,3,1,2)
        return {"rep": self.model(x)}
    
    def get_output_size(self):
        return self.out_size
    

class AnySatExtractor(nn.Module):
    def __init__(self, name, bands, **kwargs):
        super(AnySatExtractor, self).__init__()
        self.model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=False)
        self.out_size = 768
        self.name_sensor = name
        self.band_orders = bands  
        
    def forward(self, data, **kwargs):
        if self.name_sensor in ["aerial", "aerial-flair", "spot", "naip"] and len(data.shape) == 4:
            if data.shape[-1] != data.shape[-2]:
                data = data.permute(0,3,1,2)

            data_fix_order = data[:, self.band_orders]
            data_fix_order[:, self.band_orders == -1] = 0 

        elif self.name_sensor in ["s2", "s1-asc", "s1", "alos", "l7", "l8", "modis"]:
            if len(data.shape) == 3:
                data = data.unsqueeze(3).unsqueeze(3)
            elif len(data.shape) == 5 and (data.shape[-1] != data.shape[-2]):
                data = data.permute(0,1,4,2,3)

            data_fix_order = data[:,:, self.band_orders]
            data_fix_order[:,:, self.band_orders == -1] = 0 

        else:
            raise ValueError(f"Sensor {self.name_sensor} with shape {data.shape} not implemented")

        forward_data = {self.name_sensor: data_fix_order}
        
        if len(data_fix_order.shape) == 5:
            forward_data[self.name_sensor+"_dates"] = torch.linspace(0, 365, steps=data_fix_order.shape[1]).repeat(data_fix_order.shape[0], 1).to(data_fix_order.device)

        out_anysat = self.model(forward_data, patch_size=10, output="tile")
        return {"rep": out_anysat}
    
    def get_output_size(self):
        return self.out_size