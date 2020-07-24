import os

import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MultiLabelDataset(Dataset):
    """多标签数据输入."""

    def __init__(self, csv_file, root_dir, num_classes, begin=0, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.begin = begin
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = image.transpose((2, 0, 1))

        classes = pd.Series(self.landmarks_frame.iloc[idx, 1:])
        classes = ','.join(str(i)for i in classes)
        classes = classes.split(';')

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for i in range(len(classes)):
            label[int(classes[i])-self.begin] = 1

        if self.transform:
            image = self.transform(image)

        return image, label, img_name


dataset = MultiLabelDataset(csv_file='Train_label.csv', root_dir='train',
                            num_classes=29, begin=1, transform=transforms)
dataset_sizes = len(dataset)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
