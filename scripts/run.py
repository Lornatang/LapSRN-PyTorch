import os

# Prepare dataset
# Create multiscale dataset
os.system("python ./create_multiscale_dataset.py --images_dir ../data/TB291/original --output_dir ../data/TB291/LapSRN/original")
# Split image
os.system("python ./prepare_dataset.py --images_dir ../data/TB291/LapSRN/original --output_dir ../data/TB291/LapSRN/train --image_size 128 --step 64")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/TB291/LapSRN/train --valid_images_dir ../data/TB291/LapSRN/valid --valid_samples_ratio 0.1")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx4_lmdb --upscale_factor 4")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx8_lmdb --upscale_factor 8")

os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx4_lmdb --upscale_factor 4")
os.system("python ./create_lmdb_dataset.py --images_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx8_lmdb --upscale_factor 8")
