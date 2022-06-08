from src import *
from pathlib import Path
import pytorch_lightning as pl
import yaml
import sys
import importlib


config = {
    'datamodule': {
        'path': Path('dataset'),
        'batch_size': 25
    },
    'trainer': {
        'max_epochs': 10,
        'enable_checkpointing': False,
        'overfit_batches': 0
    },
    'logger': None,
    'callbacks': None
}


def train(config):
    dm =  MNISTDataModule(**config['datamodule'])
    module = MNISTModule(cfg=config)
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(module, dm)
    trainer.save_checkpoint('final.ckpt')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if config_file:
            with open(config_file) as conf_data:
                loaded_config = yaml.safe_load(conf_data)
            deep_update(config, loaded_config)
    print(config)
    train(config)
