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
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import LapSRN


def main() -> None:
    # Initialize the super-resolution model
    model = LapSRN().to(config.device)
    print("Build LapSRN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load LapSRN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Read LR image and HR image
        hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0

        if config.upscale_factor == 8:
            lr_image = imgproc.imresize(hr_image, 1 / 8)
        elif config.upscale_factor == 4:
            lr_image = imgproc.imresize(hr_image, 1 / 4)
        elif config.upscale_factor == 2:
            lr_image = imgproc.imresize(hr_image, 1 / 2)
        else:
            raise ValueError(f"Not support `upscale_factor={config.upscale_factor}`, Please use `2`, `4` and `8`.")

        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            srx2_y_tensor, srx4_y_tensor, srx8_y_tensor = model(lr_y_tensor)

        # cal PSNR and save image
        if config.upscale_factor == 8:
            total_psnr += 10. * torch.log10(1. / torch.mean((srx8_y_tensor - hr_y_tensor) ** 2))
            sr_y_image = imgproc.tensor2image(srx8_y_tensor, range_norm=False, half=True)
        elif config.upscale_factor == 4:
            total_psnr += 10. * torch.log10(1. / torch.mean((srx4_y_tensor - hr_y_tensor) ** 2))
            sr_y_image = imgproc.tensor2image(srx4_y_tensor, range_norm=False, half=True)
        elif config.upscale_factor == 2:
            total_psnr += 10. * torch.log10(1. / torch.mean((srx2_y_tensor - hr_y_tensor) ** 2))
            sr_y_image = imgproc.tensor2image(srx2_y_tensor, range_norm=False, half=True)
        else:
            raise ValueError(f"Not support `upscale_factor={config.upscale_factor}`, Please use `2`, `4` and `8`.")

        sr_y_image = np.clip(sr_y_image.astype(np.float32) / 255.0, 0.0, 1.0)
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

    print(f"PSNR: {total_psnr / total_files:4.2f}dB.\n")


if __name__ == "__main__":
    main()
