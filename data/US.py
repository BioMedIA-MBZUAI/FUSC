"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class US(Dataset):
    def __init__(self, csv_path,img_folder_path, transform= None):
        super(US,self).__init__()
        self.df = pd.read_csv(csv_path,sep=';')
        self.df = self.df[self.df.Plane != "Other"]
        self.img_folder_path = img_folder_path
        self.transform = transform
        self.classes = ["Fetal abdomen", "Fetal brain",	"Fetal femur",	"Fetal thorax",	"Maternal cervix"]
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        temp = self.df.iloc[index]
        path = os.path.join(self.img_folder_path,temp['Image_name']+".png",)
        sample = Image.open(path)
        # target = np.where(self.classes == temp['Plane'])
        target = self.classes.index(temp['Plane'])

        if self.transform is not None:
            sample = self.transform(sample)
        out = {'image': sample, 'target': target}
        return out



