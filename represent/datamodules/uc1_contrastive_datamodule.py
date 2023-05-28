import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from represent.tools.utils_uc1 import read_image
from kornia import augmentation as augs
from torch import nn
from sklearn.model_selection import train_test_split

class DataLoader():
    """
    THIS CLASS ORCHESTRATES THE TRAINING, VALIDATION, AND TEST DATA GENERATORS
    """
    def __init__(self,
                 database_dir,
                 input_type=1,
                 im_size=224,
                 is_reduced=True,
                 batch_size=2,
                 num_workers=1,
                 data_filter='s1_asc_sell'):

        tr_image_list = DataLoader.load_data(database_dir,data_filter,months=["08","09","10","11"])
        val_image_list = DataLoader.load_data(database_dir,data_filter,months=["12","07"])

        train_dataset = DataReader(tr_image_list, im_size,input_type,is_reduced)
        valid_dataset = DataReader(val_image_list, im_size,input_type,is_reduced)

        self.dataloaders = {}

        self.dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                                           shuffle=True, num_workers=num_workers, pin_memory=True)
        self.dataloaders['valid'] = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                           shuffle=True, num_workers=num_workers, pin_memory=True)
    def get_data_loader(self, type):
        return self.dataloaders[type]

    @staticmethod
    def train_val_dataset(dataset, val_split=0.20):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        train_data = Subset(dataset, train_idx)
        valid_data = Subset(dataset, val_idx)
        return train_data, valid_data


    @staticmethod
    def load_data(directory: str,data_filter:str, months=["1","02","03"]):
        """Load each cube, reduce its dimensionality and append to array.

        Args:
            directory (str): Directory to either train or test set
        Returns:
            [type]: A list with spectral curve for each sample.
        """
        concatanated=[]
        for month in months:
            all_files = np.array(
                sorted(
                    glob.glob(os.path.join(directory+'/*/{}/*/{}/'.format(month,data_filter), "*.tif")),
                )
            )
            concatanated.append(all_files)

        return np.concatenate(concatanated)



class DataReader(Dataset):
    def __init__(self, image_list, image_size,input_type,reduced_channel):

        self.input_type=input_type
        self.image_list=image_list
        self.reduced_channel=reduced_channel
        self.crop=augs.RandomCrop((image_size,image_size),p=1,keepdim=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image_name = self.image_list[index]
        img=read_image(image_name,input_type=self.input_type,is_cut=False,is_log=False,reduced_channel=self.reduced_channel)
        img=np.transpose(img,(-1,0,1))#.astype(np.float32)
        img=self.crop(torch.from_numpy(img))
        return img
