# %%
from modules.dataloader import CMRCineDataModule, CMR2DDataModule
from modules.utils import plot_batch, cine_to_3D, bbox
import argparse
import sys
sys.argv=['']
del sys
import os

PATH = os.path.dirname(os.path.abspath(__file__)) + "\dataset_3D_crop"

# set your own path here, eg, '/home/bme001/20180883/data/mnms2/sorted/SA/PerDisease' (Linux style path)
default_config = {
    'dataset_path': PATH,
    'run_name': "cmr_DDPM_11042023",
    'epochs': 50,
    'log_interval': 100,
    'batch_size' : 8,
    'image_size' : 128,
    'num_workers' : 0,  # default 8, windows cannot handle this
    'device' : "cpu",
    'lr' : 3e-4,
    'noise_steps' : 500,
    'beta_start':1e-4,
    'beta_end': 0.01,
    }

# set key-value pairs from command line 
parser = argparse.ArgumentParser()
for keys in default_config:
    parser.add_argument('--'+keys, default=default_config[keys], type= type(default_config[keys]))
args = parser.parse_args()

# CMR2DDataModule or CMRCineDataModule class
data = CMRCineDataModule(
        data_dir=args.dataset_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_val_ratio=0.8,
        num_workers=args.num_workers,
    )
data.prepare_data()
data.setup()
print('batch_size = {}'.format(args.batch_size))
dataloader = data.train_dataloader()
# print('number of images is {}'.format(len(dataloader)))
print('number of batch in train_dataloader is {}'.format(len(dataloader)))

# %%
# visualize a batch of images
plot_batch(dataloader)
# cine_to_3D(args.dataset_path)
# crop_data = bbox(str(args.dataset_path.replace('_3D', '')))
# crop_data.crop_image()

# %%
# test the training loop
from tqdm import tqdm
import logging
for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            labels = images['location'][:,2].to(args.device)
            images = images['image']['data'].squeeze(dim=-1).to(args.device   )
            images = images.to(args.device   )
            print(images.shape)
            if len(images.shape)<4:
                print(images)
# %%
