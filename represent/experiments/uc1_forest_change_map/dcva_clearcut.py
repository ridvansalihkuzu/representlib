# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Ridvan Salih Kuzu, Sudipan Saha
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm.auto import tqdm
import shutil
from skimage import morphology
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
from represent.tools.utils_uc1 import PreProcess, read_image,plot_resulting_map,interpolate_dataframe
from represent.models.uc1_resnet_dcva import init_dcva_model
from skimage import filters



class DeepChangeVectorAnalysis():

    def __init__(self, args, input_type, vector_layer_list, output_layer_list, object_min_size,morphology_size,is_saturate,top_saturate, step_size_area_analysis = 512):
        self.args=args
        self.input_type=input_type
        self.step_size_area_analysis=step_size_area_analysis
        self.vector_layer_list=vector_layer_list
        self.output_layer_list=output_layer_list
        self.object_min_size=object_min_size
        self.morphology_size=morphology_size
        self.is_saturate=is_saturate
        self.top_saturate=top_saturate
        self.label_threshold=130

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def execute(self):

        temp_dir=self.args.out_dir + '/temp/'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        try:
            label_image = read_image(self.args.ground_truth_path, is_satellite=False,is_cut=False)
            cloud_mask_image = label_image.copy()

            try: forest_mask_image = read_image(self.args.forest_mask_path,is_satellite=False,is_cut=False)
            except: forest_mask_image=np.ones(label_image.shape)
            forest_mask_1d_array = (forest_mask_image.astype(bool)).ravel()
            forest_idx = np.argwhere(forest_mask_1d_array == True)

            label_image[label_image>self.label_threshold]=1
            label_image[label_image != 1] = 0
            label_1d_array = (label_image.astype(bool)).ravel()

            cloud_mask_image[cloud_mask_image > -1] = 1
            cloud_mask_image[cloud_mask_image != 1] = 0
            cloud_mask_1d_array = (cloud_mask_image.astype(bool)).ravel()
            cloud_idx=np.argwhere(cloud_mask_1d_array==True)

            filter_idx=np.intersect1d(forest_idx,cloud_idx)
            label_image=label_image*forest_mask_image*cloud_mask_image

            if self.input_type==1:
                layer_wise_feature_extractor_S1 = init_dcva_model(self.args.model_s1_dir,self.vector_layer_list,self.args.ssl)
                pre_change_image = read_image(self.args.pre_s1_path,input_type=self.input_type,top_saturate=self.top_saturate)
                post_change_image =read_image(self.args.post_s1_path,input_type=self.input_type,top_saturate=self.top_saturate)
            elif self.input_type==2:
                layer_wise_feature_extractor_S2 = init_dcva_model(self.args.model_s2_dir,self.vector_layer_list,self.args.ssl)
                pre_change_image = read_image(self.args.pre_s2_path,input_type=self.input_type,top_saturate=self.top_saturate)
                post_change_image = read_image(self.args.post_s2_path,input_type=self.input_type,top_saturate=self.top_saturate)
            else: #self.input_type==3:
                layer_wise_feature_extractor_S1 = init_dcva_model(self.args.model_s1_dir,self.vector_layer_list,self.args.ssl)
                pre_change_image_1 = read_image(self.args.pre_s1_path,input_type=1,top_saturate=self.top_saturate)
                post_change_image_1 = read_image(self.args.post_s1_path,input_type=1,top_saturate=self.top_saturate)

                layer_wise_feature_extractor_S2 = init_dcva_model(self.args.model_s2_dir,self.vector_layer_list,self.args.ssl)
                pre_change_image_2 = read_image(self.args.pre_s2_path,input_type=2,top_saturate=self.top_saturate)
                post_change_image_2 = read_image(self.args.post_s2_path,input_type=2,top_saturate=self.top_saturate)
                pre_change_image=np.concatenate([pre_change_image_1,pre_change_image_2],-1)
                post_change_image = np.concatenate([post_change_image_1, post_change_image_2], -1)


            pre_change_image = pre_change_image * cloud_mask_image * forest_mask_image
            post_change_image = post_change_image * cloud_mask_image * forest_mask_image


            #print(pre_change_image.shape)
            #print(post_change_image.shape)


        except Exception as e:
            print(e)
        #    #sys.exit('Cannot find the images in this directory or cannot read the images')

        pre_change_image_original_shape = pre_change_image.shape
        if pre_change_image_original_shape[0] < pre_change_image_original_shape[1]:  ##code is written in a way s.t. it expects row>col
            pre_change_image = np.swapaxes(pre_change_image, 0, 1)
            post_change_image = np.swapaxes(post_change_image, 0, 1)


        if not self.args.rerun:
            patch_id=0
            pbar = tqdm( total=int(pre_change_image.shape[0]*pre_change_image.shape[1]/(self.step_size_area_analysis*self.step_size_area_analysis)),position=0, leave=True)

            for row_iter in range(0, pre_change_image.shape[0], self.step_size_area_analysis):
                for col_iter in range(0, pre_change_image.shape[1], self.step_size_area_analysis):

                    patch_id=patch_id+1
                    pbar.update(1)
                    pbar.set_postfix({'patch processed': patch_id})

                    temp_file_name_1 = temp_dir + '/temp_result_{}_{}_1.mat'.format(str(row_iter), str(col_iter))
                    temp_file_name_2 = temp_dir + '/temp_result_{}_{}_2.mat'.format(str(row_iter), str(col_iter))
                    row_end = min((row_iter + self.step_size_area_analysis), pre_change_image.shape[0])
                    col_end = min((col_iter + self.step_size_area_analysis), pre_change_image.shape[1])
                    pre_change_sub_image = pre_change_image[row_iter:row_end, col_iter:col_end, :]
                    post_change_sub_image = post_change_image[row_iter:row_end, col_iter:col_end, :]

                    if self.input_type==1:
                        self._dcva(pre_change_sub_image, post_change_sub_image, layer_wise_feature_extractor_S1, temp_file_name_1)
                    elif self.input_type==2:
                        self._dcva(pre_change_sub_image, post_change_sub_image, layer_wise_feature_extractor_S2, temp_file_name_2)
                    elif self.input_type == 3:
                        self._dcva(pre_change_sub_image[:, :, 0:-4], post_change_sub_image[:, :, 0:-4], layer_wise_feature_extractor_S1, temp_file_name_1)
                        self._dcva(pre_change_sub_image[:, :, -4:], post_change_sub_image[:, :, -4:], layer_wise_feature_extractor_S2, temp_file_name_2)


        patch_id = 0
        pbar = tqdm(total=int(pre_change_image.shape[0] * pre_change_image.shape[1] / (self.step_size_area_analysis * self.step_size_area_analysis)),position=0, leave=True)
        detected_change_map_1 = np.zeros((pre_change_image.shape[0], pre_change_image.shape[1]))
        detected_change_map_2 = np.zeros((pre_change_image.shape[0], pre_change_image.shape[1]))

        for row_iter in range(0, pre_change_image.shape[0], self.step_size_area_analysis):
            for col_iter in range(0, pre_change_image.shape[1], self.step_size_area_analysis):

                patch_id = patch_id + 1
                pbar.update(1)
                pbar.set_postfix({'patch reloaded': patch_id})

                temp_file_name_1 = temp_dir + '/temp_result_{}_{}_1.mat'.format(str(row_iter), str(col_iter))
                temp_file_name_2 = temp_dir + '/temp_result_{}_{}_2.mat'.format(str(row_iter), str(col_iter))

                row_end = min((row_iter + self.step_size_area_analysis), pre_change_image.shape[0])
                col_end = min((col_iter + self.step_size_area_analysis), pre_change_image.shape[1])

                if self.input_type==1:
                    detected_change_map_1[row_iter:row_end, col_iter:col_end] = sio.loadmat(temp_file_name_1)['detectedChangeMapThisSubArea']
                elif self.input_type==2:
                    detected_change_map_2[row_iter:row_end, col_iter:col_end] = sio.loadmat(temp_file_name_2)['detectedChangeMapThisSubArea']
                else:
                    detected_change_map_1[row_iter:row_end, col_iter:col_end] = sio.loadmat(temp_file_name_1)['detectedChangeMapThisSubArea']
                    detected_change_map_2[row_iter:row_end, col_iter:col_end] = sio.loadmat(temp_file_name_2)['detectedChangeMapThisSubArea']


        ##Normalizing the detected Change Map
        if self.input_type==1:
            detected_change_map_normalized = (detected_change_map_1 - np.nanmin(detected_change_map_1)) / (np.nanmax(detected_change_map_1) - np.nanmin(detected_change_map_1))
        elif self.input_type == 2:
            detected_change_map_normalized = (detected_change_map_2 - np.amin(detected_change_map_2)) / (np.nanmax(detected_change_map_2) - np.nanmin(detected_change_map_2))
        else:
            #detected_change_map_normalized_1 = (detected_change_map_1 - np.nanmin(detected_change_map_1)) / (np.nanmax(detected_change_map_1) - np.nanmin(detected_change_map_1))
            #detected_change_map_normalized_2 = (detected_change_map_2 - np.nanmin(detected_change_map_2)) / (np.nanmax(detected_change_map_2) - np.nanmin(detected_change_map_2))
            detected_change_map_normalized=0.5*(detected_change_map_1 + detected_change_map_2)

            detected_change_map_normalized = (detected_change_map_normalized - np.nanmin(detected_change_map_normalized)) / (np.nanmax(detected_change_map_normalized) - np.nanmin(detected_change_map_normalized))

        detected_change_map_normalized=np.nan_to_num(detected_change_map_normalized)
        ### Reading the forest mask

        cdMap_best = np.zeros(detected_change_map_normalized.shape, dtype=bool)
        max_ss = 0
        sensitivity_list = []
        specifity_list = []
        idx_list = []
        accuracy_list = []
        f1_list = []

        otsuThreshold = filters.threshold_li(detected_change_map_normalized)
        Q99 = np.percentile(detected_change_map_normalized, 99)
        steps = np.arange(0.0 * otsuThreshold, Q99, (Q99 - 0.0 * otsuThreshold) / self.args.threshold_steps)
        #steps = range(10, 500, int(500/self.args.threshold_steps))
        is_best = False
        pbar = tqdm(enumerate(steps), total=len(steps), desc="Calculating Correct Rate when SSL weights: {} ..".format(self.args.ssl),position=0, leave=True)
        for idx, scl in pbar:
            scaling_factor = scl / 1

            cdMap = (detected_change_map_normalized > scaling_factor)
            cdMap = morphology.remove_small_objects(cdMap, min_size=self.object_min_size)
            cdMap = morphology.binary_closing(cdMap, morphology.disk(self.morphology_size))
            #cdMap[forest_mask_image[:,:,0] == 0] = 0
            if pre_change_image_original_shape[0] < pre_change_image_original_shape[1]:  ##Conformity to row>col
                cdMap = np.swapaxes(cdMap, 0, 1)
                detected_change_map_normalized = np.swapaxes(detected_change_map_normalized, 0, 1)

            cd_map_1d_array = cdMap.astype(bool).ravel()
            confusion_matrix_estimated = confusion_matrix(y_true=label_1d_array[filter_idx], y_pred=cd_map_1d_array[filter_idx])

                # getting details of confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
            true_negative, false_positive, false_negative, true_positive = confusion_matrix_estimated.ravel()

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (precision * recall) / (precision + recall)

            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
            avg_ss = (specificity + sensitivity) / 2
            if avg_ss > max_ss:
                max_ss = avg_ss
                cdMap_best = cdMap

            pbar.set_postfix({'sensitivity ': sensitivity, 'specificity ': specificity,'accuracy ': accuracy,'avg': avg_ss,'max_avg': max_ss})

                # return scl,i_correct,d_correct
            sensitivity_list.append(sensitivity)
            specifity_list.append(specificity)
            idx_list.append(scaling_factor)  # TODO
            accuracy_list.append(accuracy)
            f1_list.append(f1_score)

            if accuracy>0.85:
                if not is_best:
                    cdMap_best=cdMap
                is_best=True

            if sensitivity == 0: break

        result_composite = np.zeros((detected_change_map_normalized.shape[0], detected_change_map_normalized.shape[1], 3))
        result_composite[:, :, 0] = detected_change_map_normalized
        result_composite[:, :, 1] = (cloud_mask_image * forest_mask_image)[:,:,0]
        result_composite[:, :, 2] = cdMap_best

        out_map=plot_resulting_map(cdMap_best, label_image, filter_idx)
        cdMap_best = plot_resulting_map(cdMap_best, cdMap_best, filter_idx)

        result_table = pd.DataFrame(
            {'idx': idx_list, 'sensitivity': sensitivity_list, 'specificity': specifity_list,
             'accuracy': accuracy_list, 'f1': f1_list})
        result_table = result_table.sort_values(by=['idx'])
        last_row = result_table.iloc[-1].copy()
        last_row['sensitivity'] = 0
        result_table = result_table.append(last_row, ignore_index=True)
        result_table=interpolate_dataframe(result_table)

        #print(result_table)
        shutil.rmtree(temp_dir)

        return result_table,out_map,cdMap_best,result_composite


    def _dcva(self, preChangeImage, postChangeImage, model_layers, saveNormalizedChangeMapPath):


      ##Parsing options
      output_layer_numbers = np.asarray(self.output_layer_list)

      nanVar=float('nan')

      #Defining parameters related to the CNN
      sizeReductionTable=[nanVar,4,8,16,32]
      sizeReductionTable=[sizeReductionTable[0], *[sizeReductionTable[item-4] for item in self.vector_layer_list]]

      featurePercentileToDiscardTable=[nanVar,90,90,96,98]
      featurePercentileToDiscardTable = [featurePercentileToDiscardTable[0], *[featurePercentileToDiscardTable[item-4] for item in self.vector_layer_list]]

      ##filterNumberTable=[nanVar,256,512,1024,2048]
      filterNumberTable=[nanVar,64,128,256,512]  ###For Resnet-18
      filterNumberTable = [filterNumberTable[0], *[filterNumberTable[item - 4] for item in self.vector_layer_list]]


      #When operations like filterNumberForOutputLayer=filterNumberTable[outputLayerNumber] are taken, it works, as 0 is dummy and indexing starts from 1

      data1 = np.copy(preChangeImage)
      data2 = np.copy(postChangeImage)
      #Pre-change and post-change image normalization
      #if self.is_saturate:
      #    data1=SaturateImage().execute(data1, self.top_saturate)
      #    data2=SaturateImage().execute(data2, self.top_saturate)

      #Checking image dimension
      imageSize=data1.shape
      imageSizeRow=imageSize[0]
      imageSizeCol=imageSize[1]
      imageNumberOfChannel=imageSize[2]

      torch.no_grad()

      eachPatch=int(self.step_size_area_analysis/2)
      numImageSplitRow=imageSizeRow/eachPatch
      numImageSplitCol=imageSizeCol/eachPatch
      cutY=list(range(0,imageSizeRow,eachPatch))
      cutX=list(range(0,imageSizeCol,eachPatch))
      additionalPatchPixel=int(self.step_size_area_analysis/16)



      ##Checking validity of feature extraction layers
      validFeatureExtractionLayers=[1,2,3,4] ##Feature extraction from only these layers have been defined here
      for outputLayer in output_layer_numbers:
          if outputLayer not in validFeatureExtractionLayers:
              sys.exit('Feature extraction layer is not valid, valid values are 1,2,3,4')

      ##Extracting bi-temporal features
      modelInputMean=0.406

      output_layer_number=0
      for idx, output_layer_number in enumerate(output_layer_numbers):
              #outputLayerNumber=outputLayerNumbers[outputLayerIter]
              filterNumberForOutputLayer=filterNumberTable[output_layer_number]
              featurePercentileToDiscard=featurePercentileToDiscardTable[output_layer_number]
              featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
              sizeReductionForOutputLayer=sizeReductionTable[output_layer_number]
              patchOffsetFactor=int(additionalPatchPixel/sizeReductionForOutputLayer)

              timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
              timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])


              if ((imageSizeRow<eachPatch) | (imageSizeCol<eachPatch)):
                  if imageSizeRow>imageSizeCol:
                      patchToProcessDate1=np.pad(data1,[(0,0),(0,imageSizeRow-imageSizeCol),(0,0)],'symmetric')
                      patchToProcessDate2=np.pad(data2,[(0,0),(0,imageSizeRow-imageSizeCol),(0,0)],'symmetric')
                  if imageSizeCol>imageSizeRow:
                      patchToProcessDate1=np.pad(data1,[(0,imageSizeCol-imageSizeRow),(0,0),(0,0)],'symmetric')
                      patchToProcessDate2=np.pad(data2,[(0,imageSizeCol-imageSizeRow),(0,0),(0,0)],'symmetric')
                  if imageSizeRow==imageSizeCol:
                      patchToProcessDate1=data1
                      patchToProcessDate2=data2
                  #print('This image (or this subpatch) is small and hence processing in 1 step')
                   #converting to pytorch varibales and changing dimension for input to net
                  patchToProcessDate1=patchToProcessDate1-modelInputMean
                  inputToNetDate1=torch.from_numpy(patchToProcessDate1)
                  inputToNetDate1=inputToNetDate1.float()
                  inputToNetDate1 = torch.permute(inputToNetDate1, (2, 0, 1))
                  inputToNetDate1=inputToNetDate1.unsqueeze(0)
                  del patchToProcessDate1

                  patchToProcessDate2=patchToProcessDate2-modelInputMean
                  inputToNetDate2=torch.from_numpy(patchToProcessDate2)
                  inputToNetDate2=inputToNetDate2.float()
                  inputToNetDate2 = torch.permute(inputToNetDate1, (2, 0, 1))
                  inputToNetDate2=inputToNetDate2.unsqueeze(0)
                  del patchToProcessDate2

                  #running model on image 1 and converting features to numpy format
                  inputToNetDate1 = inputToNetDate1.to(self.device)
                  with torch.no_grad():
                      obtainedFeatureVals1=model_layers[output_layer_number](inputToNetDate1)
                  obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
                  obtainedFeatureVals1=obtainedFeatureVals1.data.numpy()
                  del inputToNetDate1

                  #running model on image 2 and converting features to numpy format
                  inputToNetDate2 = inputToNetDate2.to(self.device)
                  with torch.no_grad():
                      obtainedFeatureVals2=model_layers[output_layer_number](inputToNetDate2)
                  obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
                  obtainedFeatureVals2=obtainedFeatureVals2.data.numpy()
                  del inputToNetDate2

                  for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[0:imageSizeRow,\
                                                 0:imageSizeCol,processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             0:int(imageSizeRow/sizeReductionForOutputLayer),\
                                                                             0:int(imageSizeCol/sizeReductionForOutputLayer)],\
                                                                             (imageSizeRow,imageSizeCol))
                  for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[0:imageSizeRow,\
                                                 0:imageSizeCol,processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             0:int(imageSizeRow/sizeReductionForOutputLayer),\
                                                                             0:int(imageSizeCol/sizeReductionForOutputLayer)],\
                                                                             (imageSizeRow,imageSizeCol))


              if not((imageSizeRow<eachPatch) | (imageSizeCol<eachPatch)):
                  for kY in range(0,len(cutY)):
                      for kX in range(0,len(cutX)):


                          #extracting subset of image 1
                          if (kY==0 and kX==0):
                              patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kY==0 and kX!=(len(cutX)-1)):
                              patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                          elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                              patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                          elif (kX==0 and kY!=(len(cutY)-1)):
                              patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                                        (cutY[kY]+eachPatch),\
                                                             cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                              patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                        (imageSizeRow),\
                                                             cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                              patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                        (imageSizeRow),\
                                                             (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                          else:
                              patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                                        (cutY[kY]+eachPatch),\
                                                        (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                          #extracting subset of image 2
                          if (kY==0 and kX==0):
                              patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kY==0 and kX!=(len(cutX)-1)):
                              patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                          elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                              patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                             (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                          elif (kX==0 and kY!=(len(cutY)-1)):
                              patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                                        (cutY[kY]+eachPatch),\
                                                            cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                              patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                        (imageSizeRow),\
                                                             cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                          elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                              patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                        (imageSizeRow),\
                                                             (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                          else:
                              patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                                        (cutY[kY]+eachPatch),\
                                                        (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                          #print(kY)
                          #print(kX)
                          #print(patchToProcessDate1.shape)
                          #print(patchToProcessDate2.shape)

                          #converting to pytorch varibales and changing dimension for input to net
                          patchToProcessDate1=patchToProcessDate1-modelInputMean

                          inputToNetDate1 = torch.from_numpy(patchToProcessDate1)
                          del patchToProcessDate1
                          inputToNetDate1  =inputToNetDate1.float()
                          inputToNetDate1 = torch.permute(inputToNetDate1, (2, 0, 1))
                          inputToNetDate1 = inputToNetDate1.unsqueeze(0)


                          patchToProcessDate2=patchToProcessDate2-modelInputMean

                          inputToNetDate2=torch.from_numpy(patchToProcessDate2)
                          del patchToProcessDate2
                          inputToNetDate2 = inputToNetDate2.float()
                          inputToNetDate2 = torch.permute(inputToNetDate2, (2, 0, 1))
                          inputToNetDate2 = inputToNetDate2.unsqueeze(0)


                          #running model on image 1 and converting features to numpy format
                          inputToNetDate1 = inputToNetDate1.to(self.device)
                          with torch.no_grad():
                              obtainedFeatureVals1=model_layers[output_layer_number](inputToNetDate1)
                          obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
                          obtainedFeatureVals1=obtainedFeatureVals1.cpu().numpy()
                          del inputToNetDate1

                          #running model on image 2 and converting features to numpy format
                          inputToNetDate2 = inputToNetDate2.to(self.device)
                          with torch.no_grad():
                              obtainedFeatureVals2=model_layers[output_layer_number](inputToNetDate2)
                          obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
                          obtainedFeatureVals2=obtainedFeatureVals2.cpu().numpy()
                          del inputToNetDate2
                          #this features are in format (filterNumber, sizeRow, sizeCol)



                          ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
                          obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
                          obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)


                          #obtaining features from image 1: resizing and truncating additionalPatchPixel
                          if (kY==0 and kX==0):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             (eachPatch,eachPatch))

                          elif (kY==0 and kX!=(len(cutX)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                             (eachPatch,eachPatch))
                          elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals1.shape[2])],\
                                                                             (eachPatch,(imageSizeCol-cutX[kX])))
                          elif (kX==0 and kY!=(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             (eachPatch,eachPatch))
                          elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:imageSizeRow,\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals1.shape[1]),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             ((imageSizeRow-cutY[kY]),eachPatch))
                          elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals1.shape[1]),\
                                                                             (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals1.shape[2])],\
                                                                             ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                          else:
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                             (eachPatch,eachPatch))
                          #obtaining features from image 2: resizing and truncating additionalPatchPixel
                          if (kY==0 and kX==0):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             (eachPatch,eachPatch))

                          elif (kY==0 and kX!=(len(cutX)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                             (eachPatch,eachPatch))
                          elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                             (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals2.shape[2])],\
                                                                             (eachPatch,(imageSizeCol-cutX[kX])))
                          elif (kX==0 and kY!=(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             (eachPatch,eachPatch))
                          elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:imageSizeRow,\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals2.shape[1]),\
                                                                             0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                             ((imageSizeRow-cutY[kY]),eachPatch))
                          elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals2.shape[1]),\
                                                                             (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                             (obtainedFeatureVals2.shape[2])],\
                                                                             ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                          else:
                              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                                  timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                                 cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                                 resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                             (patchOffsetFactor+1):\
                                                                             (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                             (eachPatch,eachPatch))

              del  obtainedFeatureVals1,obtainedFeatureVals2
              timeVectorDifferenceMatrix=timeVector1Feature-timeVector2Feature
              #print(timeVectorDifferenceMatrix.shape)

              nonZeroVector=[]
              stepSizeForStdCalculation=min(int(imageSizeRow/2),1024)
              for featureSelectionIter1 in range(0,imageSizeRow,stepSizeForStdCalculation):
                  for featureSelectionIter2 in range(0,imageSizeCol,stepSizeForStdCalculation):
                      timeVectorDifferenceSelectedRegion=timeVectorDifferenceMatrix\
                                                         [featureSelectionIter1:(featureSelectionIter1+stepSizeForStdCalculation),\
                                                          featureSelectionIter2:(featureSelectionIter2+stepSizeForStdCalculation),
                                                          0:filterNumberForOutputLayer]
                      stdVectorDifferenceSelectedRegion=np.std(timeVectorDifferenceSelectedRegion,axis=(0,1))
                      featuresOrderedPerStd=np.argsort(-stdVectorDifferenceSelectedRegion)   #negated array to get argsort result in descending order
                      nonZeroVectorSelectedRegion=featuresOrderedPerStd[0:featureNumberToRetain]
                      nonZeroVector=np.union1d(nonZeroVector,nonZeroVectorSelectedRegion)



              modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
              modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]
              del timeVector1Feature,timeVector2Feature

              ##Normalize the features (separate for both images)
              meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))
              stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
              normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image

              meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))
              stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
              normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image

              ##feature aggregation across channels
              if idx==0:
                  timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
                  timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
              else:
                  timeVector1FeatureAggregated=np.concatenate((timeVector1FeatureAggregated,normalizedModifiedTimeVector1),axis=2)
                  timeVector2FeatureAggregated=np.concatenate((timeVector2FeatureAggregated,normalizedModifiedTimeVector2),axis=2)

      if self.is_saturate:
          timeVector1FeatureAggregated=PreProcess.execute(timeVector1FeatureAggregated, top_saturate=self.top_saturate)
          timeVector2FeatureAggregated=PreProcess.execute(timeVector2FeatureAggregated, top_saturate=self.top_saturate)

      absoluteModifiedTimeVectorDifference=np.absolute(timeVector1FeatureAggregated-timeVector2FeatureAggregated)

      #take absolute value for binary CD
      detectedChangeMap=np.linalg.norm(absoluteModifiedTimeVectorDifference,axis=(2))

      sio.savemat(saveNormalizedChangeMapPath,{"detectedChangeMapThisSubArea": detectedChangeMap})


