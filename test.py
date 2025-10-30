from dataset import create_dataset
from tqdm import tqdm
import logging
from train import create_network

logging.basicConfig(level=logging.INFO)

train_loader, val_loader = create_dataset()
model = create_network("x3d_m")

for item in tqdm(train_loader, total=3000):
    model(item['video'])
