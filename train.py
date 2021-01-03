
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision.transforms import Normalize

from network.models import model_selection
from data_list import ImageList
from mlflow import log_params, log_metrics

torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyNet(nn.Module):

    def __init__(self, args) -> None:
        super(MyNet, self).__init__()

        self.net, *_ = model_selection(modelname='xception', num_out_classes=2)
        self.net = self.net.to(device)
        self.batch = args.batch
        self.max_images = args.max_images
        self.threshold = args.threshold
        self.criterion = nn.CrossEntropyLoss()
        self.dataloader = self.get_dataloader(args)
    
    def count_parameters(self, trainable=False):
        if trainable:
            return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.net.parameters())

    def set_trainable(self, layer):
        self.net.set_trainable_up_to(True, layer)
    

    def set_optimizer(self, lr):
        self.optimizer = Adam(self.net.parameters(), lr)

    @staticmethod
    def get_dataloader(args):
        image_dir = args.image_dir

        train = open(os.path.join(image_dir, "train_dir.txt")).readlines()
        val = open(os.path.join(image_dir, "val_dir.txt")).readlines()
        test = open(os.path.join(image_dir, "test_dir.txt")).readlines()

        img_size = 299
        mean = [0.5] * 3
        std = [0.5] * 3
        normalize_transform = Normalize(mean, std)

        trainset = ImageList(train, normalize_transform, img_size)
        valset = ImageList(val, normalize_transform, img_size)
        testset = ImageList(test, normalize_transform, img_size)

        dataloader = dict()
        for split in ["train", "val", "test"]:
            dataloader[split] = DataLoader(eval(split + "set"), batch_size=1, shuffle=True)

        return dataloader
    
    def train(self, epoch):
        print("\nEpoch:", epoch)
        self.net.train()
        train_img_loss = 0
        correct_imgs = 0
        correct_vds = 0
        total_imgs = 0
        total_vds = 0
        labels = []
        video_loss = 0
        img_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.dataloader["train"]):
            inputs = inputs.squeeze(0)
            # trim to preven GPU out of memeory
            if inputs.shape[0] > self.max_images:
                inputs = inputs[:self.max_images]
            # print(inputs.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels.append(targets[:])
            targets = targets.repeat(len(inputs))
            
            outputs = self.net(inputs)
            prediction_imgs = outputs.argmax(1)
            correct_imgs += prediction_imgs.eq(targets).sum().item()
            loss = self.criterion(outputs, targets)
            loss.backward()
            img_loss += loss.item()
            # img_loss += loss

            prediction_vds = 1 if prediction_imgs.float().mean() >= self.threshold else 0
            # print(loss)
            
            correct_vds += (prediction_vds == labels[-1]).item()
            total_imgs += len(targets)
            
            if (batch_idx + 1) % self.batch != 0:
                continue
            
            # img_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_img_loss += img_loss
            # train_img_loss += img_loss.item()
            
            total_vds += len(labels)
            labels = []
            img_loss = 0

            if (batch_idx + 1) % 10 == 0:
                print("Iteration", batch_idx + 1)
                print("train_img_loss: {:.3f}".format(train_img_loss / (batch_idx + 1)))
                print("train_img_acc: {:.3f}".format(1.0 * correct_imgs / total_imgs))
                print("train_vds_acc: {:.3f}\n".format(1.0 * correct_vds / total_vds))


        log_metrics(
                {
                    "train_img_loss": train_img_loss / (batch_idx + 1),
                    "train_img_acc": 1.0 * correct_imgs / total_imgs,
                    "train_vds_acc": 1.0 * correct_vds / total_vds,
                },
                step=epoch,
            )

    def evaluate(self, split, epoch):
        self.net.eval()
        total_img_loss = 0
        correct_imgs = 0
        correct_vds = 0
        total_imgs = 0
        total_vds = 0
        labels = []
        video_loss = 0
        img_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataloader[split]):
                inputs = inputs.squeeze(0)
                # trim to preven GPU out of memeory
                if inputs.shape[0] > self.max_images:
                    inputs = inputs[:self.max_images]
                inputs = inputs.to(device)
                targets = targets.to(device)
                labels.append(targets[:])
                targets = targets.repeat(len(inputs))
                
                outputs = self.net(inputs)
                prediction_imgs = outputs.argmax(1)
                correct_imgs += prediction_imgs.eq(targets).sum().item()
                loss = self.criterion(outputs, targets)
                
                img_loss += loss.item()

                prediction_vds = 1 if prediction_imgs.float().mean() >= self.threshold else 0
                
                correct_vds += (prediction_vds == labels[-1]).item()
                total_imgs += len(targets)
                
                if (batch_idx + 1) % self.batch != 0:
                    continue

                total_img_loss += img_loss
                
                total_vds += len(labels)
                labels = []
                img_loss = 0

            log_metrics(
                    {
                        f"{split}_img_loss": total_img_loss / (batch_idx + 1),
                        f"{split}_img_acc": 1.0 * correct_imgs / total_imgs,
                        f"{split}_vds_acc": 1.0 * correct_vds / total_vds,
                    },
                    step=epoch,
                )
            
            return 1.0 * correct_vds / total_vds

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--max_images', type=int, default=200)
    parser.add_argument('--layer', type=str, default='conv3')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--model_path', type=str, default='pretrained/xception/all_raw.p')
    parser.add_argument('--output_dir', type=str, default='weights')
    args = parser.parse_args()

    model = MyNet(args)
    model.net = torch.load(args.model_path)
    model.set_trainable(args.layer)
    model.set_optimizer(args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "xception.pth")

    log_params(vars(args))
    print("Total params: {:.2f} M".format(model.count_parameters() / 1e6))
    print("Trainable params: {:.2f} M".format(model.count_parameters(trainable=True) / 1e6))

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train(epoch)

        if epoch % args.interval == 0:
            acc = model.evaluate(split="val", epoch=epoch)
            if acc > best_val_acc:
                state_dict = model.net.state_dict()
                torch.save(state_dict, path)
                best_val_acc = acc
    
    # loading the "best" model
    path = torch.load(path)
    model.net.load_state_dict(path)

    model.evaluate(split="test", epoch=None)