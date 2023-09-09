import os
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2
import numpy as np
import argparse


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.jit.load(args.model_root).to(device)
    net.eval()

    tr = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    for item in tqdm(
            os.listdir(args.data_folder),
            desc="Testing"):
        image = cv2.imread(os.path.join(args.data_folder, item), 1)
        img = tr(image).reshape((1, 3, 256, 256)).to(device)
        x = net(img)
        output = x.cpu().detach().numpy()
        numpy_image = output.transpose(0, 2, 3, 1)[0, :, :, 0]
        img_normalized = cv2.normalize(numpy_image, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(f"{args.save_folder}/{item}", img_normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Hair segmentation')
    parser.add_argument("--model_root", '-m',
                        default="/home/mohammad/Documents/Abolfaz_hair50.pt")
    parser.add_argument("--data_folder", '-v',
                        default="/home/example")
    parser.add_argument("--save_folder", '-s',
                        default="./Result")
    main(parser.parse_args())
