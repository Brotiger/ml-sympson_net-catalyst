import torch
import numpy as np
import PIL

import pickle
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms, models, datasets
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
from catalyst import dl
import math
import time
import torch.optim as optim

RESCALE_SIZE = 224
train_dir = "./journey-springfield/train/simpsons_dataset"

def scale_pixel_values(x):
    return x / 255
    
train_transforms = transforms.Compose([
    transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(scale_pixel_values),
   # transforms.Lambda(lambda x: x / 255),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
            
image_datasets = datasets.ImageFolder(train_dir, train_transforms)
    
dataset_len = len(image_datasets)
p1 = (dataset_len / 100)
train_size = math.floor(p1 * 70)
test_size = dataset_len - train_size
            
train_dataset, val_dataset = torch.utils.data.random_split(image_datasets, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128
)
    
loaders = {
    "train": train_dataloader,
    "valid": val_dataloader
}

class CustomSupervisedRunner(dl.SupervisedRunner):
    def get_loaders(self):
        train_transforms = transforms.Compose([
            transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
        image_datasets = datasets.ImageFolder(train_dir, train_transforms)
    
        dataset_len = len(image_datasets)
        p1 = (dataset_len / 100)
        train_size = math.floor(p1 * 70)
        test_size = dataset_len - train_size
            
        train_dataset, val_dataset = torch.utils.data.random_split(image_datasets, [train_size, test_size])
    
        if self.engine.is_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.engine._world_size,
                rank=self.engine.process_index,
                shuffle=False,
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.engine._world_size,
                rank=self.engine.process_index,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None
    
        train_loader = DataLoader(
            train_dataset, batch_size=128, sampler=train_sampler
        )
        valid_loader = DataLoader(
            val_dataset, batch_size=128, sampler=val_sampler
        )
        return {"train": train_loader, "valid": valid_loader}
            
if __name__ == '__main__':
    runner = dl.SupervisedRunner()

    
    
    model_resnet18 = models.resnet18(pretrained=True)
    
    for param in model_resnet18.parameters():
        param.requires_grad = False
    
    for param in model_resnet18.avgpool.parameters():
        param.requires_grad = True
        
    for param in model_resnet18.layer4.parameters():
        param.requires_grad = True
    
    for param in model_resnet18.layer3.parameters():
        param.requires_grad = True
    
    for param in model_resnet18.layer2.parameters():
        param.requires_grad = True
    
    num_classes = 42
    model_resnet18.fc = nn.Sequential(
        nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, num_classes)
        )
    ) 
    
    optimizer = optim.Adam(
        (
            {
                "params": model_resnet18.layer2.parameters(),
                "lr": 1e-5
            },
            {
                "params": model_resnet18.layer3.parameters(),
                "lr": 1e-4
            },
            {
                "params": model_resnet18.layer4.parameters(),
                "lr": 1e-3
            },
            {
                "params": model_resnet18.avgpool.parameters()
            },
            {
                "params": model_resnet18.fc.parameters()
            }
        ), 
    lr=1e-2)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
    
    start_time = time.time()
    
    runner.train(
        engine=dl.DistributedDataParallelEngine(),
        model=model_resnet18,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        callbacks=[
            dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
            dl.BackwardCallback(metric_key="loss"),
            dl.OptimizerCallback(metric_key="loss"), 
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            dl.SchedulerCallback(),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=num_classes, log_on_batch=False
            )
        ],
        loaders=loaders,
        num_epochs=35,
        verbose=True,
        logdir="logs/resnet18",
        ddp=True,
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Время выполнения: {execution_time} секунд")