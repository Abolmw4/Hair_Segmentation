from torch.utils.data import Dataset, DataLoader
import cv2
import os
import re


class MyData(Dataset):
    def __init__(self, data_dir, transfomr):
        super(MyData, self).__init__()
        self.transform = transfomr
        self.data_image = [os.path.join(data_dir, "images", item) for item in
                           os.listdir(os.path.join(data_dir, "images"))]
        self.mask_image = [os.path.join(data_dir, "masks", item) for item in
                           os.listdir(os.path.join(data_dir, "masks"))]
        self.images = sorted(self.data_image, key=lambda s: int(re.search(r'\d+', s).group()))
        self.mask = sorted(self.mask_image, key=lambda s: int(re.search(r'\d+', s).group()))

    def __len__(self):
        return len(self.data_image)

    def __getitem__(self, item):
        image = self.transform(cv2.imread(self.images[item], 1))
        # mask = self.transform(cv2.cvtColor(cv2.imread(self.mask[item]), cv2.COLOR_BGR2GRAY))
        mask = self.transform(cv2.imread(self.mask[item], cv2.IMREAD_GRAYSCALE))
        gray_image = self.transform(cv2.cvtColor(cv2.imread(self.images[item], 1), cv2.COLOR_BGR2GRAY))

        return image, mask, gray_image
