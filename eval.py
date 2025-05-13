import argparse
import os
import random
import time
import torch

from core.model import DiscrepancyEstimator

if __name__ == '__main__':
    model = DiscrepancyEstimator(from_pretrained='./ckpt/DDL_Qwen2-0.5B_grok3_polish_zh_trai_e5_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8')
    print(model)