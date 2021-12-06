import os

# Prepare dataset
os.system("python ./prepare_dataset.py --inputs_dir ../data/TB291/original --output_dir ../data/TB291/LapSRN --image_size 128 --step 64")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --inputs_dir ../data/TB291/LapSRN --valid_samples_ratio 0.1")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx4_lmdb --upscale_factor 4")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/train --lmdb_path ../data/train_lmdb/LapSRN/TB291_LRbicx8_lmdb --upscale_factor 8")

os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_HR_lmdb --upscale_factor 1")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx2_lmdb --upscale_factor 2")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx4_lmdb --upscale_factor 4")
os.system("python ./create_lmdb_dataset.py --inputs_dir ../data/TB291/LapSRN/valid --lmdb_path ../data/valid_lmdb/LapSRN/TB291_LRbicx8_lmdb --upscale_factor 8")
