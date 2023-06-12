import os
import shutil
import random
import argparse
from tqdm import tqdm

def get_filepath(folder_path: str, file_postfix:str = "") -> str:
    """
    Returns the path of the file with the given postfix in the given folder.

    Args:
        folder_path (str): Path to the folder.
        file_postfix (str): Postfix of the file.

    Returns:
        str: Path to the file. If no file with the given postfix exists, returns None.
    """
    for file in os.listdir(folder_path):
        if file_postfix in file:
            return os.path.join(folder_path, file), file.replace(file_postfix, "")
    return None, None

def copy_folder_or_file(src_path: str, output_path: str, split: str, file_postfix:str = "") -> None:
    """
    Copies a folder or file to a destination path.

    Args:
        src_path (str): Path to the source folder or file.
        dst_path (str): Path to the destination folder or file.
        file_postfix (str): Postfix of the file.

    Returns:
        None
    """
    if file_postfix:
        folder_name = os.path.basename(src_path)
        src_path, output_filename = get_filepath(src_path, file_postfix)
        dst_path = os.path.join(output_path, split, folder_name)
        os.makedirs(dst_path, exist_ok=True)

        shutil.copy(src_path, os.path.join(dst_path, output_filename))
    else:
        dst_path = os.path.join(output_path, split, os.path.basename(src_path))
        shutil.copytree(src_path, dst_path)

def split_dataset(dataset_path: str, output_path: str, ratio: str, file_postfix:str = "") -> None:
    """
    Splits a dataset into train, validation, and test sets and copies them to an output directory.

    Args:
        dataset_path (str): Path to the dataset folder.
        output_path (str): Path to the output folder.
        ratio (str): Ratio of train, validation, and test sets as comma-separated values (e.g. "60,20,20").
        file_postfix (str): Postfix of the files to be copied.

    Returns:
        None
    """

    # Check if ratio is valid (train,val,test)
    check_ratio = [int(r) for r in ratio.split(',')]
    assert len(check_ratio) == 3, 'Ratio must be in the form of "train,val,test" and integer values must be given for each split.'

    # Create output directories
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    # List all folders in dataset_path
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    # Shuffle folders randomly
    random.shuffle(folders)

    # Split folders into train, val, and test sets
    train_ratio, val_ratio, test_ratio = [int(r) for r in ratio.split(',')]
    total_ratio = train_ratio + val_ratio + test_ratio
    train_cutoff = int(len(folders) * train_ratio / total_ratio)
    val_cutoff = train_cutoff + int(len(folders) * val_ratio / total_ratio)
    train_folders = folders[:train_cutoff]
    val_folders = folders[train_cutoff:val_cutoff]
    test_folders = folders[val_cutoff:]

    # Copy folders to output_path
    for split, split_folder in [('train', train_folders), ('val', val_folders), ('test', test_folders)]:
        for folder in tqdm(split_folder, desc=f'Copying {split} set'):
            copy_folder_or_file(os.path.join(dataset_path, folder), output_path, split, file_postfix)
       

if __name__ == '__main__':
    """usage: dataset_splitter.py [-h] dataset_path output_path ratio

    Split dataset into train, validation, and test sets.

    positional arguments:
    dataset_path  Path to dataset folder
    output_path   Path to output folder

    optional arguments:
    --ratio         Ratio of train, validation, and test sets as comma-separated values (e.g. "60,20,20")
    --file_postfix  Postfix of the files to be copied
    -h, --help    show this help message and exit

    Example usage:
    python dataset_splitter.py /path/to/dataset /path/to/output 60,20,20
    """


    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets.')
    parser.add_argument('dataset_path', type=str, help='Path to dataset folder')
    parser.add_argument('output_path', type=str, help='Path to output folder')
    parser.add_argument('--ratio', default='60,20,20', type=str, help='Ratio of train, validation, and test sets as comma-separated values (e.g. "60,20,20")')
    parser.add_argument('--file_postfix', type=str, default='', help='Postfix of the files to be copied. If not specified, all files will be copied.')
    args = parser.parse_args()

    split_dataset(args.dataset_path, args.output_path, args.ratio, args.file_postfix)
