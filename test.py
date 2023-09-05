import os
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = torch.jit.load("/home/mohammad/Documents/Hair_segmentation_Abolfazl/weight/new_weights/Abolfaz_hair50.pt").to(device)
net.eval()

tr = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
for item in tqdm(os.listdir("/home/mohammad/Documents/Gender_classification_train_version/dataset/test_data/female"), desc="Testing"):
    image = cv2.imread(f"/home/mohammad/Documents/Gender_classification_train_version/dataset/test_data/female/{item}", 1)
    img = tr(image).reshape((1, 3, 256, 256)).to(device)
    x = net(img)
    output = x.cpu().detach().numpy()
    numpy_image = output.transpose(0, 2, 3, 1)[0, :, :, 0]
    img_normalized = cv2.normalize(numpy_image, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"/home/mohammad/Documents/Hair_segmentation_Abolfazl/newRES/{item}", img_normalized)
