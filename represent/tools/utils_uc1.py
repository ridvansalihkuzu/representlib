#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Author: Ridvan Salih Kuzu - Sudipan Saha.

"""

import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import rasterio
from scipy.interpolate import interp1d
#import seaborn as sns
#sns.set_style("whitegrid")
import pandas as pd



class PreProcess():
    ##Defines code for image adjusting/pre-processing

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    @staticmethod
    def execute( input_data, is_log=False, is_cut=False,top_saturate=1):
        if input_data.ndim == 2:
            inputSarLog = np.log(input_data)
            inputSarLogNormalized = PreProcess._normalize_band(inputSarLog,is_cut=is_cut,top_saturate=top_saturate)
            return inputSarLogNormalized
        elif input_data.ndim == 3:
            numBands = input_data.shape[2]
            if (numBands > input_data.shape[0]) or (numBands > input_data.shape[1]):
                sys.exit('Input for the SAR preprpcessing is expected as R*C*Bands')
            sarPreprocessed = copy.deepcopy(input_data)
            if is_log: sarPreprocessed = np.log(input_data)
            inputSarBandLogNormalized = PreProcess._normalize_band(sarPreprocessed, (0, 1),is_cut=is_cut,top_saturate=top_saturate)
            return inputSarBandLogNormalized

        else:
            sys.exit('Input for the SAR preprpcessing expects 2 or 3 band image')

    @staticmethod
    def _normalize_band(inputMapToNormalize,axis=None,is_cut=False,top_saturate=1):
        inputMapBand = inputMapToNormalize#.astype(float)
        inputMapBand=np.nan_to_num(inputMapBand,nan=0,neginf=np.nanmin(inputMapBand[inputMapBand != -np.inf]),posinf=np.nanmax(inputMapBand[inputMapBand != np.inf]))
        mi=np.percentile(inputMapBand, top_saturate,axis=axis)
        ma=np.percentile(inputMapBand, 100-top_saturate,axis=axis)
        inputMapNormalizedBand = (inputMapBand - mi ) / (np.finfo(float).eps + ma - mi)
        if is_cut:
            inputMapNormalizedBand[inputMapNormalizedBand > 1] = 1
        return inputMapNormalizedBand



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def read_image(image_path, input_type=0, is_satellite=True, is_cut=False,is_log=True,reduced_channel=True, top_saturate=2):
        if is_satellite:
            image_tif = rasterio.open(image_path)
            image = image_tif.read().astype(np.float32)
            image = np.transpose(image, (1, 2, 0))
            if input_type == 1:
                image = image[:, :, [1, 0]]
                image = PreProcess.execute(image, is_log,is_cut,top_saturate)
            elif input_type == 2:
                if reduced_channel:
                    image = image[:, :, [0, 1, 2, 6]]
                else:
                    image = image[:, :, :-1]
                image = PreProcess.execute(image, False, is_cut,top_saturate)


        else:
            image_tif = rasterio.open(image_path)
            image = image_tif.read().astype(np.float32)
            image = np.transpose(image, (1, 2, 0))

        return image

def plot_resulting_map(predicted_map, ground_truth_map,filter_idx=None):
    out = np.zeros((predicted_map.shape[0], predicted_map.shape[1], 3))
    for i in range(predicted_map.shape[0]):
        for j in range(predicted_map.shape[1]):
            ref = int(ground_truth_map[i, j])
            res = int(predicted_map[i, j])
            if ref > 0 and res > 0:
                out[i, j, :] = (255, 255, 255)
            elif ref == 0 and res == 0:
                out[i, j, :] = (0, 0, 0)
            elif ref == 0 and res > 0:  # False Positives
                out[i, j, :] = (255, 0, 255)
            elif ref > 0 and res == 0:  # False Negatives
                out[i, j, :] = (0, 255, 255)
            else:
                print('Something Wrong')

    def _filter_out(data,filter_idx):
        shape = data.shape
        in_flat = np.ravel(data)
        out_flat = np.zeros(in_flat.shape)
        out_flat[filter_idx] = in_flat[filter_idx]
        data = out_flat.reshape(*shape)
        return data

    if filter_idx is not None:
        out[:, :, 0] = _filter_out(out[:, :, 0], filter_idx)
        out[:, :, 1] = _filter_out(out[:, :, 1], filter_idx)
        out[:, :, 2] = _filter_out(out[:, :, 2], filter_idx)

    return out

def calculate_eer(far, frr):
    """ Returns the most optimal FAR and FRR values """

    far_optimum = far[np.nanargmin(np.absolute((frr - far)))]
    frr_optimum = frr[np.nanargmin(np.absolute((frr - far)))]

    return far_optimum, frr_optimum


def plot_DET_ROC(far, frr, far_optimum, frr_optimum, figure_name):
    """ Plots a DET curve with the most suitable operating point based on threshold values"""
    fig = plt.figure()
    lw = 2
    # Plot the DET curve based on the FAR and FRR values
    area = np.abs(np.trapz(far, x=frr))
    plt.plot([0, 1.01], [0, 1.01], color='blue', lw=lw, linestyle='--')
    # plt.plot(far, frr, color='red', linewidth=lw, label='DET Curve')
    # Plot the optimum point on the DET Curve
    plt.plot(far, frr, color='red', linewidth=lw, label='ROC Curve (AUC = %0.3f)' % area)

    plt.plot(far_optimum, frr_optimum, "ko", label="Suitable Operating Point")

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Sensitivity')
    plt.ylabel('Specificity')
    plt.title('ROC Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    fig.savefig(figure_name, dpi=fig.dpi)


def plot_DET_ROC_all(figure_name, title, labels, *args):
    fig, ax = plt.subplots(figsize=(5, 5))  # Set equal width and height
    lw = 2

    ax.plot([0, 1.01], [0, 1.01], color='black', lw=lw, linestyle='dotted')

    colors = ['brown', 'royalblue', 'forestgreen', 'dimgrey','orange']

    for i, (far, frr, far_optimum, frr_optimum, acc, f1) in enumerate(args):
        # Calculate the trapezoidal area under the curve
        area = np.abs(np.trapz(far, x= frr))
        ax.plot(far, frr, color=colors[i], linewidth=lw, linestyle='-',
                label=f'ROC Curve for {labels[i]}\n(AUC = {area:.3f}, F1 = {f1:.3f})')
        ax.plot(far_optimum, frr_optimum, "o", color='black', markersize=8,
                label="Suitable Operating Point" if i == len(args)-1 else None)
        ax.plot(far_optimum, frr_optimum, "o", color=colors[i], markersize=8)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Sensitivity', fontsize=12)
    ax.set_ylabel('Specificity', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.set_xticks([x / 10.0 for x in range(11)])
    ax.set_yticks([y / 10.0 for y in range(11)])

    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)  # Add grid to the plot
    plt.tight_layout()
    plt.savefig(figure_name, dpi=144)


def process_and_plot(result_tables, title, labels, output_file):
    results = []
    for table in result_tables:
        #table = table.sort_values(by=['idx'])
        #table=interpolate_dataframe(table)
        #table = table.sort_values(by=['idx'],ascending=False)

        #last_row = table.iloc[-1].copy()
        #last_row['sensitivity'] = 0
        #table = table.append(last_row, ignore_index=True)
        far=table.to_numpy()[:, 1]
        frr=table.to_numpy()[:, 2]
        far_optimum, frr_optimum = calculate_eer(far, frr)
        acc = table[table.sensitivity == far_optimum].accuracy.values[0]
        f1 = np.max(table.f1.values)
        #f1 = table[table.sensitivity == far_optimum].f1.values[0]
        results.append((far, frr, far_optimum, frr_optimum, acc, f1))

    plot_DET_ROC_all(output_file, title, labels, *results)


def interpolate_array(y, l=250):
    # Get x-values from array indices
    x = np.arange(len(y))

    # Interpolate y-values using cubic spline
    f = interp1d(x, y, kind='linear')

    # Generate new x-values and interpolate corresponding y-values
    new_x = np.linspace(x.min(), x.max(), l)
    new_y = f(new_x)

    return new_y


def interpolate_dataframe(df,l=250):
    # Get x-values from DataFrame index
    x = df.index.values

    # Initialize a new DataFrame to hold interpolated values
    new_df = pd.DataFrame(index=np.linspace(x.min(), x.max(), l))

    # Interpolate each column using cubic spline
    for col in df.columns:
        y = df[col].values
        f = interp1d(x, y, kind='linear')
        new_df[col] = f(new_df.index)

    return new_df



