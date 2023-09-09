import argparse
import torch.cuda
from model import AolfazlNet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optimizer
import torch.nn as nn
from data import MyData
from loss import *
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--data_root_train", '-t',
                        default="/home/mohammad/Documents/Hair_segmentation_Abolfazl/dataset/train")
    parser.add_argument("--data_root_val", '-v',
                        default="/home/mohammad/Documents/Hair_segmentation_Abolfazl/dataset/val")
    parser.add_argument("--data_root_test", '-i',
                        default="/home/mohammad/Documents/Hair_segmentation_Abolfazl/dataset/test")
    parser.add_argument("--weight", '-w',
                        default="/home/mohammad/Documents/Hair_segmentation_Abolfazl/weight")
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    MyNet = AolfazlNet(input_channel=3, out_channels=1, pretrained=True).to(device)
    criteraion = BCELoss2d()
    optimiz = optimizer.Adam(MyNet.parameters(), lr=0.0001)
    tr = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    train_data = MyData(data_dir=args.data_root_train, transfomr=tr)
    tr_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    validation_data = MyData(data_dir=args.data_root_val, transfomr=tr)
    val_data = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    epoch_val_loss, epoch_train_loss, IOU_VAL, IOU_TR = [], [], [], []

    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        iou_val = 0.0
        iou_train = 0.0

        for (img, msk, gray_image) in tqdm(tr_data, desc=f"Train | Epoch {epoch + 1}/{args.epochs}", leave=True):
            image = img.to(device)
            mask = msk.to(device)
            gry = gray_image.to(device)
            optimiz.zero_grad()
            pred = MyNet(image)
            loss = criteraion(pred, mask)
            l2_norm = sum(p.pow(2.0).sum() for p in MyNet.parameters())
            loss1 = 2.0 * loss + 2.5 * (1 - dice_coeff(pred, mask)) + 0.0008 * l2_norm
            train_loss += loss1.item()
            loss1.backward()
            optimiz.step()
        epoch_train_loss.append(train_loss / len(tr_data))
        with torch.no_grad():
            MyNet.eval()
            for (img, msk, gray_image) in tqdm(val_data, desc=f"Val | Epoch {epoch + 1}/{args.epochs}", leave=True):
                image = img.to(device)
                mask = msk.to(device)
                gry = gray_image.to(device)
                pred = MyNet(image)
                loss = criteraion(pred, mask)
                l2_norm = sum(p.pow(2.0).sum() for p in MyNet.parameters())
                loss2 = 2.0 * loss + 2.5 * (1 - dice_coeff(pred, mask)) + 0.0008 * l2_norm
                val_loss += loss2.item()

            epoch_val_loss.append(val_loss / len(val_data))
            IOU_VAL.append(iou_val / len(val_data))
        print(f'loss_train: {epoch_train_loss[-1]} | loss_val: {epoch_val_loss[-1]}\n')
        if (epoch + 1) % 2 == 0:
            model_scripted = torch.jit.script(MyNet)  # Export to TorchScript
            model_scripted.save(
                f'{args.weight}/Abolfaz_hair{epoch + 1}.pt')  # Save
    return epoch_train_loss, epoch_val_loss, args.epochs


if __name__ == "__main__":
    t_loss, v_loss, epochs = main()
    x = [i + 1 for i in range(epochs)]
    plt.plot(np.array(x), np.array(t_loss), label='train')
    plt.plot(np.array(x), np.array(v_loss), label='val')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()
