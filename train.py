import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.model import DiscrepancyEstimator
from core.dataset import CustomDataset
from core.loss import calculate_DPO_loss, calculate_DDL_loss
from core.metrics import AUROC, AUPR
from core.trainer import Trainer
from accelerate import Accelerator
from peft import LoraConfig