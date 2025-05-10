import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from .model import DiscrepancyEstimator
from .dataset import CustomDataset
from .loss import calculate_DPO_loss, calculate_DDL_loss
from accelerate import Accelerator


class Trainer():
    def train(self,
              accelerator: Accelerator,
              model: DiscrepancyEstimator,
              train_dataset: CustomDataset,
              loss_fn,
              learning_rate: float=1e-4,):
        start_time = time.time()
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)