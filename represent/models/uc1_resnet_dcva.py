"""
Code Author: Ridvan Salih Kuzu, Sudipan Saha.
"""

import torch
import torch.nn as nn


def init_dcva_model(model_dir, vector_layer_list, is_ssl):
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(accelerator)

    net = torch.load(model_dir, map_location=device)

    if is_ssl: #TODO
        l1 = list(net.children())
    else:
        l1 = list(net.children())

    net_list = []
    for i in vector_layer_list:
        if i > 0:
            net_list.append(nn.Sequential(*l1[:i]))

    for net_i in net_list:
        net_i.eval()
        net_i.requires_grad = False

    nanVar = float('nan')
    layer_wise_feature_extractor_function = [nanVar, *net_list]
    return layer_wise_feature_extractor_function
