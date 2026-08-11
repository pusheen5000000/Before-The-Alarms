import os
import shutil
import pandas as pd

# Path configuration based on your structure
ROOT_DIR = "1"  # Your top-level folder
CSV_PATH = os.path.join(ROOT_DIR, "UrbanSound8K.csv")

OUTPUT_GUNSHOT_DIR = "dataset/gunshot"
OUTPUT_OTHER_DIR = "dataset/other"  # Set to None if you only want gunshots

def extract_gunshots(csv_path, root_dir, gunshot_dir, other_dir=None):
    os.makedirs(gunshot_dir, exist_ok=True)
    if other_dir:
        os.makedirs(other_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    gunshot_count = 0
    other_count = 0

    print("Extracting audio files...")

    for _, row in df.iterrows():
        slice_name = row['slice_file_name']
        fold = f"fold{row['fold']}"
        class_id = int(row['classID'])

        # Constructs path directly: 1/foldX/filename.wav
        src_path = os.path.join(root_dir, fold, slice_name)

        if not os.path.exists(src_path):
            print(f"Warning: File not found -> {src_path}")
            continue

        # ClassID 6 corresponds to gunshot
        if class_id == 6:
            dst_path = os.path.join(gunshot_dir, slice_name)
            shutil.copy(src_path, dst_path)
            gunshot_count += 1
        elif other_dir is not None:
            dst_path = os.path.join(other_dir, slice_name)
            shutil.copy(src_path, dst_path)
            other_count += 1

    print("\nExtraction Complete!")
    print(f"Copied {gunshot_count} gunshot files to '{gunshot_dir}'")
    if other_dir:
        print(f"Copied {other_count} non-gunshot files to '{other_dir}'")

if __name__ == "__main__":
    extract_gunshots(CSV_PATH, ROOT_DIR, OUTPUT_GUNSHOT_DIR, OUTPUT_OTHER_DIR)