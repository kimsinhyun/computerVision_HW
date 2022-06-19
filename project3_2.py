import csv
from turtle import forward
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary   import summary
from torchmetrics import F1Score
import json
import os
import pandas as pd
import cv2
from PIL import Image 
import visdom
import glob

def loss_tracker(vis, loss_plot, loss_value, num):

    vis.line(
                X=num,
                Y=loss_value,
                win = loss_plot,
                update='append'
             )

class block(nn.Module):
    def __init__(
        self, in_channels, out_channel, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, out_channel, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(
            out_channel,
            out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class MyModel(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(MyModel, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.build_layer(
            block, layers[0], out_channel=64, stride=1
        )
        self.layer2 = self.build_layer(
            block, layers[1], out_channel=128, stride=2
        )
        self.layer3 = self.build_layer(
            block, layers[2], out_channel=256, stride=2
        )
        self.layer4 = self.build_layer(
            block, layers[3], out_channel=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def build_layer(self, block, num_residual_blocks, out_channel, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channel * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channel * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channel * 4),
            )

        layers.append(
            block(self.in_channels, out_channel, identity_downsample, stride)
        )

        self.in_channels = out_channel * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channel))

        return nn.Sequential(*layers)


def parse_file_number(col):
    return int(col.split(".")[0])
def parse_file_number_test(col):
    col = col.replace('./test_data\\','')
    return col


class MyDataset(Dataset) :
    def __init__(self,meta_path,root_dir,transform=None,pre_transform=None, mode="train") :
        super().__init__()
        #===============meta data===============
        if mode == "train":
            self.mode = 'train'
            with open(meta_path, 'r') as file:
                temp_meta_data = json.load(file)
            meta = pd.json_normalize(temp_meta_data['annotations'])
            meta['file_name'] = meta['file_name'].apply(parse_file_number)
            meta = meta.sort_values("file_name").reset_index(drop=True)
        
            self.root_dir = root_dir

            meta['file_name'] = meta['file_name'].map(lambda x :  self.root_dir + '/' + str(x) +'.jpg')
            self.X = []
            loop = tqdm(list(meta['file_name']), total=len(meta['file_name']), leave=True)

            for i, X in enumerate(loop):
                try:
                    self.X.append(pre_transform(Image.open(X).convert("RGB")))
                except:
                    pass
            self.y = meta['category']
            self.transform = transform

        elif mode == "test":
            self.mode = 'test'
            meta = pd.DataFrame()
            meta['file_name'] = glob.glob(root_dir+"/*.jpg")
            meta = meta.sort_values("file_name").reset_index(drop=True)
            self.root_dir = root_dir
            self.X = []
            loop = tqdm(list(meta['file_name']), total=len(meta['file_name']), leave=True)
            for i, X in enumerate(loop):
    #             print(i)
                try:
                    self.X.append(pre_transform(Image.open(X).convert("RGB")))
                except:
                    pass
            self.y = meta['file_name'].apply(parse_file_number_test)
            self.transform = transform
        
    def __len__(self) :
        return len(self.X)
    
    def __getitem__(self,idx) :
        if  self.mode == 'train':
            X, y = self.transform(self.X[idx]), int(self.y[idx])
            return X, torch.tensor(y)
        elif self.mode == 'test':
            X ,y = self.transform(self.X[idx]), self.y[idx]
            return X ,y

def train() :
    #======================make dataset======================
    batch = 20
    mode = 'train' 
    train_data_dir = "./train_data"
    meta_path = "./answer.json"

    pre_transformer = transforms.Compose([
        transforms.Resize((400,400)),
        transforms.CenterCrop((224,224)),
    ])

    transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_data = MyDataset(meta_path, train_data_dir, transform=transformer,pre_transform=pre_transformer, mode="train")

    train_loader = DataLoader(
        train_data, batch_size=batch)
    
    #======================init RestNet101======================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel(block, [3,4,23,3], 3, 80).to(device)

    #======================make tracker======================
    # vis = visdom.Visdom()
    # vis.close(env="main")
    # loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    
    #======================start training======================
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.003,momentum=0.9)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    epochs = 26
    # epochs = 1
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        lr_sche.step()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for i, (inputs, labels) in enumerate(loop):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward & backward & optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    #         lr_sche.step()

            # print statistics
            running_loss += loss.item()
            if i % 30 == 29:    # print every 30 mini-batches
                # loss_tracker(loss_plt, torch.Tensor([running_loss/30]), torch.Tensor([i + epoch*len(train_loader) ]))
                loop.desc = "valid epoch[{}/{}], Loss= {}".format(epoch + 1, epochs, running_loss/30)
                running_loss = 0.0
    torch.save(model.state_dict(), "./model.pth")
    print('Training Finished')
    

def get_model(model_name, checkpoint_path):
    model = model_name(block, [3,4,23,3], 3, 80)
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    model = get_model(model_name,checkpoint_path)

    pre_transformer = transforms.Compose([
        transforms.Resize((400,400)),
        transforms.CenterCrop((224,224)),
    ])

    transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # Create training and validation datasets
    test_datasets = MyDataset(meta_path, data_dir,transform=transformer,pre_transform=pre_transformer, mode="test")

    batch_size = 20
    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Inference
    result = []
    for images, filename in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'filename': filename[i],
                'class': preds[i].item()
            })

    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    # Save to csv
    with open('./result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])


def main() :
    # train()
    test()    


if __name__ == '__main__':
    main()