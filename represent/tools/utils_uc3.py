
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

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


def anomaly_score_reports(df_results_organic, df_results_synthetic, min_percentile=80, step_resolution=1,
                          anomaly_precision=0, per_building=True):
    '''
        THIS FUNCTION MANAGES THE ACCURACY REPORTING FOR EACH ANOMALY TYPE
        :param df_results_organic: pandas data-frame with colmuns [footprint_id, loss, anomaly]
                                                                          where "loss" is the anomaly score (higher loss means higher likelihood of being anomaly),
                                                                          where "anomaly" is always 0 since their anomaly conditions is unknown
        :param df_results_synthetic: pandas data-frame with colmuns [footprint_id, loss, anomaly]
                                                                          where "loss" is the anomaly score (higher loss means higher likelihood of being anomaly),
                                                                          where "anomaly" is 0 for nunperturbed points and 1,2,3 for ["trend", "noise", "step"] perturbed points

        :param min_percentile: minimum percentile for thresholding anomalous points or buildings
        :param step_resolution: steps between percentile 100 and min_percentile.
        :param anomaly_precision: it is valid only if 'per_building=True'.
                                  It is the min percentage of anomalous points in order to label a building as anomalous.
                                  If it is 0, even a single point in entire PSPs is sufficient to label the building as anomalous.
        :param per_building: if it is true, the score reports are given in terms of building number, otherwise in terms of PSPs

        :return: reports as dataframe,
                 index in dataframe showing upper_whisker threshold,
                 overall accuracy of anomaly detection
    '''

    def _filter_by_anomalous_buildings(df_results, threshold, point_precision=0.0):
        df_building = df_results
        for name, group in df_results.groupby('footprint_id'):
            anoms = sum(group.loss > threshold)
            total = len(group.loss)
            if anoms / total <= point_precision:
                df_building = df_building[df_building.footprint_id != name]

        return df_building

    def _stats_per_anomaly_buildings(dframe, anom_type, threshold, point_precision=0.0):
        anom_idx = dframe.anomaly == anom_type
        df_synthetic_anom_sub_psp = dframe[anom_idx]
        total_building_num = len(df_synthetic_anom_sub_psp.groupby('footprint_id'))
        df_synthetic_anom_sub_build = _filter_by_anomalous_buildings(df_synthetic_anom_sub_psp, threshold,
                                                                     point_precision)
        anom_building_num = len(df_synthetic_anom_sub_build.groupby('footprint_id'))

        return anom_building_num, total_building_num

    def _stats_per_anomaly_PSP(dframe, anom_type, threshold):

        df_synthetic_anom = dframe[dframe.loss >= threshold]
        anom_psp_num = sum(df_synthetic_anom.anomaly == anom_type)
        total_psp_num = sum(dframe.anomaly == anom_type)

        return anom_psp_num, total_psp_num

    def _anomaly_statistics(pred):
        Q1 = np.percentile(pred, 25)
        med = np.median(pred)
        Q3 = np.percentile(pred, 75)
        IQR = Q3 - Q1
        LowerWhisker = np.max([pred.min(), Q1 - 1.5 * IQR])
        UpperWhisker = np.min([pred.max(), Q3 + 1.5 * IQR])
        return (LowerWhisker, UpperWhisker)

    def _rearrange_per_percentile(organic_data, synthetic_data, percentile):
        set1 = np.asarray((df_results_synthetic.anomaly == 0).values == True).nonzero()[0]

        arr = np.asarray((df_results_synthetic.anomaly > 0).values == True).nonzero()
        labels = df_results_synthetic[df_results_synthetic.anomaly > 0].anomaly.values
        test_size = (400 - 4 * percentile) / 100
        set3, set2, _, _ = train_test_split(arr[0], labels, test_size=test_size, random_state=42)

        df1 = df_results_synthetic.filter(items=set1, axis=0)
        df2 = df_results_synthetic.filter(items=set2, axis=0)
        df3 = df_results_organic.filter(items=set3, axis=0)
        df = pd.concat([df1, df2, df3])
        return df

    class_weights = [0.792, 0.038, 0.17]
    d, u = _anomaly_statistics(df_results_organic.loss.values)

    percentile_steps = np.arange(100 - 1 * step_resolution, min_percentile, -1 * step_resolution)

    trend_array = np.zeros(percentile_steps.shape)
    noise_array = np.zeros(percentile_steps.shape)
    step_array = np.zeros(percentile_steps.shape)
    overall_acc_array = np.zeros(percentile_steps.shape)
    overall_acc_weighted_array = np.zeros(percentile_steps.shape)
    loss_th_array = np.zeros(percentile_steps.shape)

    tqdm_object = tqdm(percentile_steps, total=len(percentile_steps))

    for idx, p_step in enumerate(tqdm_object):

        tqdm_object.set_postfix(percentile=p_step)

        th_o = np.percentile(df_results_organic.loss.values, p_step)
        df_organic_norms = df_results_organic[df_results_organic.loss < th_o]
        df_synthetic_norms = df_results_synthetic[df_results_synthetic.footprint_id.isin(df_organic_norms.footprint_id)]

        try:
            df_synthetic_norms = _rearrange_per_percentile(df_organic_norms, df_synthetic_norms, p_step)
        except Exception as e:
            print("WARNING: Step resolution parameter={} might be too small.".format(step_resolution))

        th_s = np.percentile(df_synthetic_norms.loss.values, p_step)

        if per_building:
            TP_trend, num_trend = _stats_per_anomaly_buildings(df_synthetic_norms, 1, th_s,
                                                               point_precision=anomaly_precision)
            TP_noise, num_noise = _stats_per_anomaly_buildings(df_synthetic_norms, 2, th_s,
                                                               point_precision=anomaly_precision)
            TP_step, num_step = _stats_per_anomaly_buildings(df_synthetic_norms, 3, th_s,
                                                             point_precision=anomaly_precision)

        else:
            TP_trend, num_trend = _stats_per_anomaly_PSP(df_synthetic_norms, 1, th_s)
            TP_noise, num_noise = _stats_per_anomaly_PSP(df_synthetic_norms, 2, th_s)
            TP_step, num_step = _stats_per_anomaly_PSP(df_synthetic_norms, 3, th_s)

        trend_array[idx] = TP_trend / num_trend
        noise_array[idx] = TP_noise / num_noise
        overall_acc_array[idx] = (TP_trend + TP_noise + TP_step) / (num_trend + num_noise + num_step)
        overall_acc_weighted_array[idx] = class_weights[0] * (TP_trend / num_trend) + class_weights[1] * (
                    TP_noise / num_noise) + class_weights[2] * (TP_step / num_step)
        step_array[idx] = TP_step / num_step

        loss_th_array[idx] = th_o

    # find the index of outlier from the array
    index = np.absolute(loss_th_array - u).argmin()

    df = pd.DataFrame(np.concatenate([
        np.expand_dims(percentile_steps, -1),
        np.expand_dims(trend_array, -1),
        np.expand_dims(noise_array, -1),
        np.expand_dims(step_array, -1),
        np.expand_dims(overall_acc_array, -1),
        np.expand_dims(overall_acc_weighted_array, -1)
    ], 1),
        columns=["step_array", "trend_acc", "noise_acc", "step_acc", "overall_acc", "weighted_acc"])

    return df, index, overall_acc_array[-1]


def frame_anomaly_table(losses,unique_ids, footprints, anomalies):

    losses=np.mean(losses,1,keepdims=True)
    if losses.shape[-1]==1:
        df = pd.DataFrame({"unique_id":unique_ids,
                           "footprint_id":footprints,
                           "loss":losses[:,0,0],
                            "anomaly":anomalies}).astype({'unique_id':'int32','footprint_id':'int32','anomaly':'int32'})
    elif losses.shape[-1]==3:
        df = pd.DataFrame({"unique_id": unique_ids,
                           "footprint_id": footprints,
                           "loss": np.mean(losses[:,0,:],-1),
                           "loss_0": losses[:, 0, 0],
                           "loss_1": losses[:, 0, 1],
                           "loss_2": losses[:, 0, 2],
                           "anomaly": anomalies}).astype({'unique_id': 'int32', 'footprint_id': 'int32', 'anomaly': 'int32'})

    #df = df.sort_values(["loss"])

    return df

