# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Ridvan Salih Kuzu, Sudipan Saha
"""
import os

import optuna
import joblib
import gc
from optuna.samplers import TPESampler
import warnings
import matplotlib.pyplot as plt
from represent.tools.utils_uc1 import calculate_eer, plot_DET_ROC
from represent.experiments.uc1_forest_change_map.args_windstorm import ArgumentsDCVA
from represent.experiments.uc1_forest_change_map.dcva_windstorm import DeepChangeVectorAnalysis
import numpy as np


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore")


def main(args):


    def objective(trial):
        gc.collect()

        print(f"\nTRIAL NUMBER: {trial.number}\n")
        is_saturate = args.is_saturate
        input_type = args.input_type

        top_saturate = trial.suggest_categorical('top_saturate', args.top_saturate)
        object_min_size = trial.suggest_categorical('object_min_size', args.object_min_size)
        morphology_size = trial.suggest_categorical('morphology_size', args.morphology_size)

        output_layers = trial.suggest_categorical('output_layers', args.output_layers)
        end_step_list = args.layers_process# [5, 6, 7, 8]
        output_layers = [output_layers]


        dcva = DeepChangeVectorAnalysis(args, input_type, end_step_list, output_layers, object_min_size,morphology_size, is_saturate, top_saturate)
        result_table, out_map, change_map,result_map = dcva.execute()

        far_optimum, frr_optimum = calculate_eer(result_table.to_numpy()[:, 1], result_table.to_numpy()[:, 2])
        area = np.trapz(result_table.to_numpy()[:, 1], x=result_table.to_numpy()[:, 2])

        log_args = [args.out_dir, area, input_type, args.ssl, output_layers, end_step_list, object_min_size,morphology_size, is_saturate, top_saturate]
        result_table.to_json('{}/resultComposite_e_{:0.3f}_in_{}_s_{}_l_{}_e_{}_o_{}_m_{}_s_{}_{}.json'.format(*log_args))
        plot_DET_ROC(result_table.to_numpy()[:, 1], result_table.to_numpy()[:, 2], far_optimum, frr_optimum,
                          '{}/resultComposite_e_{:0.3f}_in_{}_s_{}_l_{}_e_{}_o_{}_m_{}_s_{}_{}.png'.format(*log_args))

        #plt.imsave('{}/OUT_MAP_e_{:0.3f}_in_{}_s_{}_l_{}_e_{}_o_{}_m_{}_s_{}_{}.png'.format(*log_args),out_map.astype(float) / 255)
        #plt.imsave('{}/CHANGE_MAP_e_{:0.3f}_in_{}_s_{}_l_{}_e_{}_o_{}_m_{}_s_{}_{}.png'.format(*log_args),change_map.astype(float)/ 255)
        plt.imsave('{}/RESULT_MAP_e_{:0.3f}_in_{}_s_{}_l_{}_e_{}_o_{}_m_{}_s_{}_{}.png'.format(*log_args),result_map.astype(float))

        print("INFO: Optimum correctly detected AUC {:0.3f}".format(area))
        return area

    optuna_logs = [args.out_dir, args.input_type, args.ssl, args.layers_process]
    optuna_log_dir = '{}/model_best_i_{}_s_{}_l_{}.pkl'.format(*optuna_logs)
    study = optuna.create_study(sampler=TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30,
                                                                   interval_steps=10), direction='maximize')
    if os.path.isfile(optuna_log_dir):
        study = joblib.load(optuna_log_dir)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    joblib.dump(study, optuna_log_dir)




if __name__ == '__main__':
    args = ArgumentsDCVA().parseOptions()
    main(args)
