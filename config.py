# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn as cudnn

# ==============================================================================
# General configuration
# ==============================================================================
torch.manual_seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
upscale_factor = 4
mode = "train"
exp_name = "baseline"

# ==============================================================================
# Training configuration
# ==============================================================================
if mode == "train":
    # Dataset
    # Image format
    train_image_dir = "data/TB291/LapSRN/train"
    valid_image_dir = "data/TB291/LapSRN/valid"

    # LMDB format
    train_lrbicx2_lmdb_path = f"data/train_lmdb/LapSRN/TB291_LRbicx2_lmdb"
    train_lrbicx4_lmdb_path = f"data/train_lmdb/LapSRN/TB291_LRbicx4_lmdb"
    train_lrbicx8_lmdb_path = f"data/train_lmdb/LapSRN/TB291_LRbicx8_lmdb"
    train_hr_lmdb_path = f"data/train_lmdb/LapSRN/TB291_HR_lmdb"

    valid_lrbicx2_lmdb_path = f"data/valid_lmdb/LapSRN/TB291_LRbicx2_lmdb"
    valid_lrbicx4_lmdb_path = f"data/valid_lmdb/LapSRN/TB291_LRbicx4_lmdb"
    valid_lrbicx8_lmdb_path = f"data/valid_lmdb/LapSRN/TB291_LRbicx8_lmdb"
    valid_hr_lmdb_path = f"data/valid_lmdb/LapSRN/TB291_HR_lmdb"

    image_size = 128
    batch_size = 64
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 150

    # SGD optimizer parameter (less training and low PSNR)
    model_optimizer_name = "sgd"
    model_lr = 1e-5
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # Adam optimizer parameter (faster training and better PSNR)
    # model_optimizer_name = "adam"
    # model_lr = 1e-5
    # model_betas = (0.9, 0.999)

    # Optimizer scheduler parameter
    lr_scheduler_name = "StepLR"
    lr_scheduler_step_size = 50
    lr_scheduler_gamma = 0.5

    print_frequency = 100

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/last.pth"
