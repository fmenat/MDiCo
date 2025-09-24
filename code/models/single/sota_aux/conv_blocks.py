from torch import nn

class Conv2DBlock(nn.Module):
    '''
    Conv2D block with batch normalization and dropout layer
    '''
    def __init__(self, in_channels, n_filters, k_size, dropout, padding: int=0, pooling: bool = False, pool_size: int = 2, stride_pool: int = 2):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(n_filters)
        self.pooling = nn.MaxPool2d(pool_size, stride=stride_pool, padding=0) if pooling else nn.Identity()
        self.drop_layer = nn.Dropout2d(dropout)

        self.block = nn.Sequential(self.conv, self.act, self.bn, self.pooling, self.drop_layer)
    
    def forward(self, x):
        return self.block(x)