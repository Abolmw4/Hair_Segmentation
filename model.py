import torch.nn as nn
import torch


class AolfazlNet(nn.Module):
    def __init__(self, input_channel: int = 3, out_channels: int = None,
                 pretrained: bool = True):
        super(AolfazlNet, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channels
        self.pretrained = pretrained
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=self.input_channel,
                                    out_channels=self.out_channel, init_features=32,
                                    pretrained=self.pretrained)

    def forward(self, input):
        return self.model(input)


