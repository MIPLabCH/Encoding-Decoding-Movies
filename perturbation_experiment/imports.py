# -*- coding: utf-8 -*-

"""
This file contains all necessary imports.
"""

### Shared dependencies ###
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import nibabel as nib

### dataset.py dependencies ###
import imageio
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize

### model.py dependencies ###
from torchvision.transforms import Resize
from torchmetrics.image import TotalVariation
from torchvision.models import vgg16

### visualisation.py dependencies ###
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import scipy.stats as stats
from scipy.stats import permutation_test as perm_test
from sklearn.metrics import explained_variance_score #still needed ?