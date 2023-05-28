import torch
import torch.nn as nn
from torchvision import models

resnet_dict = {'ResNet18': models.resnet18, 'ResNet34': models.resnet34, 'ResNet50': models.resnet50,
               'ResNet101': models.resnet101, 'ResNet152': models.resnet152}



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResNetFc(nn.Module):
    def __init__(self, resnet_name, model_path, use_bottleneck=True, bottleneck_dim=256, class_num=1000, num_channels=3):
        super(ResNetFc, self).__init__()
        try:
            model_resnet = resnet_dict[resnet_name](pretrained=True)
        except:
            model_resnet = resnet_dict[resnet_name](pretrained=False)
            model_resnet.load_state_dict(torch.load(model_path))

        if num_channels == 3:
            self.conv1 = model_resnet.conv1
        else:
            self.conv1 = torch.nn.Conv2d(
                num_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

            #self.conv1 = model_resnet.conv1
            #layer0Weight = model_resnet.conv1.weight
            #layer0Weight = layer0Weight[:, 0:1, :, :]
            #layer0Weight = layer0Weight.expand(-1, numberOfImageChannels, -1,
            #                                   -1)  ##here -1 means not changing the shape of that dimension
            #self.conv1.weight = torch.nn.Parameter(layer0Weight)

        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
                                            self.layer2, self.layer3, self.layer4, self.avgpool)
        self.useBottleneck = use_bottleneck
        if self.useBottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.useBottleneck:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y

    def output_num(self):
        return self.__in_features

