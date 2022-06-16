import csv
from turtle import forward
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class MyModel(nn.Module) :
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #TODO: Make your own model
        self.in_channels = in_channels
        self.out_channles = out_channels

    def forward(self,x) :
        #TODO:
        pass


class MyDataset(Dataset) :

    def __init__(self,meta_path,root_dir,transform=None) :
        super().__init__()
        pass
    def __len__(self) :
        pass
    def __getitem__(self,idx) :
        pass


def train() :
    #TODO: Make your own training code

    # You SHOULD save your model by
    # torch.save(model.state_dict(), './checkpoint.pth') 
    # You SHOULD not modify the save path
    pass


def get_model(model_name, checkpoint_path):
    
    model = model_name()
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    model = get_model(model_name,checkpoint_path)

    data_transforms = {
        'train' :"YOUR_DATA_TRANSFORM_FUNCTION" , 
        'test': "YOUR_DATA_TRANSFORM_FUNCTION"
    }

    # Create training and validation datasets
    test_datasets = MyDataset(meta_path, data_dir, data_transforms['mode'])

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
            writer.writerow([res['filename'], result['class']])


def main() :
    pass


if __name__ == '__main__':
    main()