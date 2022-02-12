import os

# Prepare dataset
os.system("python ./create_multiscale_dataset.py --images_dir ../data/TB291/original --output_dir ../data/TB291/LapSRN/original")
os.system("python ./prepare_dataset.py --images_dir ../data/TB291/LapSRN/original --output_dir ../data/TB291/LapSRN/train --image_size 128 --step 64 --num_workers 10")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/TB291/LapSRN/train --valid_images_dir ../data/TB291/LapSRN/valid --valid_samples_ratio 0.1")
