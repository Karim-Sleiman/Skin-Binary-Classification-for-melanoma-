import os
import numpy as np
import torch
import monai
from monai.data import Dataset, DataLoader
import random
from Conventional.Preprocessing import transforms
from monai.data import pad_list_data_collate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)


def get_data(Data_path,mode):
    Data_path = os.path.join(Data_path,mode)
    Dataset=[]
    if mode=="test":
        for image_path in os.listdir(Data_path):
            Dataset.append({"image_path":os.path.join(Data_path,image_path),"image":os.path.join(Data_path,image_path)})
    else :
        for class_file in os.listdir(Data_path):
            for image_path in os.listdir(os.path.join(Data_path,class_file)):
                Dataset.append({"image_path":os.path.join(Data_path,class_file,image_path),"image":os.path.join(Data_path,class_file,image_path),"label":1 if class_file=="nevus" else 0})
    return Dataset


def get_loader(Data_path,transforms=transforms,mode = 'train',shuffle=True,batch_size=12,seed=42):
    set_seed(seed)
    data=get_data(Data_path,mode)
    Data_set=Dataset(data,transforms)
    Data_loader=DataLoader(Data_set,batch_size=batch_size,shuffle=shuffle)
    return Data_loader