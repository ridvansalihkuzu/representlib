# Represent Project
This document is part of the RepreSent project proposal that was accepted in January 2022 and responded to ESA ITT for AI4EO Challenges – Non-Supervised Learning (AO/1-10552/21/I-DT). The project is performed under ESA Contract No. 4000137253/22/I-DT.
## Introduction

The main scope of the RepreSent project is to design, implement and validate artificial intelligence (AI) non-supervised techniques that will allow the use of the Copernicus Sentinel data. These techniques are developed for:

- Fusion of Sentinel sensors (e.g., Sentinel-1 and Sentinel-2)
- Single sensor classification (multispectral or SAR Sentinel sensors)
- Change detection (e.g., Sentinel-1 or Sentinel-2)
- Image time series analysis (e.g., Sentinel-2)

The software is implemented as an open source library. The validation  is done on five use cases (UC) related to:

- UC1) Forest disturbance monitoring
- UC2) Automated Land Cover mapping
- UC3) Anomaly detection in long time series of PS-P InSAR
- UC4) Cloud detection and removal
- UC5) Forest biomass estimation

The resulting datasets of the project is to be distributed freely to the AI4EO community.

## Dataset Folder Structure

The dataset provided as part of the RepreSent project follows a specific folder structure to organize the files and code. Here's an overview of the directory structure:

- notebooks
- represent
    - callbacks
    - config
    - data
    - datamodules
    - experiments
        - uc1_contrastive_learning
        - uc1_forest_change_map
        - uc2_settlement_evaluation.py
        - uc3_building_anomaly_detection
        - uc3_benchmark
            - ganf
            - maxdiv
            - dense_autoencoder.py
        - uc3_lstm_autoencoder.py
        - uc4_odc.py
        - uc4_resnet.py
    - losses
    - models
        - moco.py
        - simclr_resnet.py
        - uc1_byol.py
        - uc1_pixel_level_contrastive_learning.py
        - uc1_resnet_base.py
        - uc1_resnet_dcva.py
        - uc2_maml.py
        - uc2_segmentation_resnet.py
        - uc2_supervised_resnet.py
        - uc3_benchmark
            - ganf
            - maxdiv
        - uc3_lstm_autoencoder.py
        - uc4_odc.py
        - uc4_resnet.py
    - tools
