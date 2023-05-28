import torch
import torch.nn as nn
import torchvision.models as models


class DoubleResNetSimCLR(nn.Module):
    def __init__(
        self,
        base_model,
        s1_channels=2,
        s2_channels=13,
        emb_dim=128,
    ):
        super(DoubleResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }

        self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=emb_dim)
        self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=emb_dim)

        self.backbone1.conv1 = torch.nn.Conv2d(
            s1_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        self.backbone2.conv1 = torch.nn.Conv2d(
            s2_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        dim_mlp1 = self.backbone1.fc.in_features
        dim_mlp2 = self.backbone2.fc.in_features

        # add mlp projection head
        self.backbone1.fc = nn.Sequential(nn.Linear(dim_mlp1, dim_mlp1), nn.ReLU(), self.backbone1.fc)
        self.backbone2.fc = nn.Sequential(nn.Linear(dim_mlp2, dim_mlp2), nn.ReLU(), self.backbone2.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x, concat: bool = False):
        x1 = self.backbone1(x["s1"])
        x2 = self.backbone2(x["s2"])

        if concat:
            z = torch.cat([x1, x2], dim=1)
            z = self.fc(z)
            return z
        else:
            return {"s1": x1, "s2": x2}

    def load_trained_state_dict(self, weights, drop_head: bool = True):
        """load the pre-trained backbone weights"""

        # remove the MLP projection heads
        if drop_head:
            for k in list(weights.keys()):
                if k.startswith(("backbone1.fc", "backbone2.fc")):
                    del weights[k]

        log = self.load_state_dict(weights, strict=False)
        if drop_head:
            assert all([k.startswith(("backbone1.fc", "backbone2.fc")) for k in log.missing_keys])

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if not name.startswith(("backbone1.fc", "backbone2.fc")):
                param.requires_grad = False
            else:
                param.requires_grad = drop_head


if __name__ == "__main__":
    from represent.models.simclr_resnet import DoubleResNetSimCLR

    net = DoubleResNetSimCLR("resnet50", s1_channels=2, s2_channels=13, emb_dim=128)
    weights = torch.load("represent/weights/simclr_resnet50.pth", map_location=torch.device("cpu"))
    net.load_trained_state_dict(weights["state_dict"], drop_head=False)
    encoder_s1 = net.backbone1
    encoder_s2 = net.backbone2
