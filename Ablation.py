import os
import random
import shutil
from pathlib import Path

LABELED_FOLDER = 'data/slapi/labeled'
UNLABELED_FOLDER = 'data/slapi/unlabeled'
TRAIN_FOLDER = 'data/slapi/train'
VAL_FOLDER = 'data/slapi/val'
TEST_FOLDER = 'data/slapi/test'
META_TRAIN_FOLDER = 'data/UCF-101/train'
META_VAL_FOLDER = 'data/UCF-101/val'
AL_FOLDER = 'data/slapi/labeled/active-learning'

"""V0: Increasingly larger dataset size
"""
#for shots in [1, 2, 3, 5, 10, 15, 25]:
#    os.system(f'python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh {shots} -av V0.{shots}')

"""V1
"""
#for i in range(10):
#    os.system('rm -R data/slapi/train/*')
#    os.system('rm -R data/slapi/val/*')
#    os.system('rm -R data/slapi/test/*')
#    os.system(f'python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -av V1.{i}')


"""V2: Optical flow preprocessing
"""
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -of -av V2")

"""V3: Leave one out traing and testing
"""
def create_data_splits(test_video_code, train_ratio=0.6, val_ratio=0.4, split_file='data/slapi/SPLIT', include_file='data/slapi/INCLUDE', LABELED_FOLDER="data/slapi/LABELED", TRAIN_FOLDER="data/slapi/TRAIN", VAL_FOLDER="data/slapi/VAL", TEST_FOLDER="data/slapi/TEST"):
    # Read the INCLUDE file and create a set of video codes to be included
    with open(include_file, 'r') as f:
        included_video_codes = set(line.strip() for line in f.readlines())

    # Check if TRAIN_FOLDER, VAL_FOLDER, and TEST_FOLDER exist, create them if not
    for folder in [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Filter the video_codes based on the included_video_codes set
    video_codes = [vc for vc in os.listdir(LABELED_FOLDER) if vc in included_video_codes]
    
    # Separate test_video_code from other video codes
    video_codes.remove(test_video_code)
    test_video_codes = [test_video_code]
    
    # Organize videos by class
    class_videos = {}
    for video_code_list, is_test in [(video_codes, False), (test_video_codes, True)]:
        for video_code in video_code_list:
            video_code_path = os.path.join(LABELED_FOLDER, video_code)
            for class_name in os.listdir(video_code_path):
                if class_name == 'ALL':  # Skip the 'ALL' folder
                    continue

                class_path = os.path.join(video_code_path, class_name)
                if class_name not in class_videos:
                    class_videos[class_name] = {'trainval': [], 'test': []}
                for video_file in os.listdir(class_path):
                    class_videos[class_name]['test' if is_test else 'trainval'].append(os.path.join(class_path, video_file))

    # Check if train, val, and test folders are empty
    if any([len(os.listdir(TRAIN_FOLDER)), len(os.listdir(VAL_FOLDER)), len(os.listdir(TEST_FOLDER))]):
        print("The train, val, or test folder is not empty. Aborting.")
        return
    
    # Clear the SPLIT file if it already exists
    with open(os.path.join(split_file), 'w') as f:
        pass

    # Create train, validation, and test splits
    split_data = {'train': TRAIN_FOLDER, 'val': VAL_FOLDER, 'test': TEST_FOLDER}
    with open(os.path.join(split_file), 'w') as f:
        for class_name, split_videos in class_videos.items():
            for split, videos in split_videos.items():
                random.shuffle(videos)
                n = len(videos)
                if split == 'trainval':
                    train_count = max(1, int(n * train_ratio))
                    
                    for idx, video_path in enumerate(videos):
                        if idx < train_count:
                            split = 'train'
                        else:
                            split = 'val'
                
                # Write the split information to the text file
                f.write(f"{video_path} {split}\n")

                # Copy the video to the corresponding split folder
                dest_folder = os.path.join(split_data[split], class_name)
                Path(dest_folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, os.path.join(dest_folder, os.path.basename(video_path)))

for i, code in enumerate([417, 867, 702, 814]):
    os.system('rm -R data/slapi/train/*')
    os.system('rm -R data/slapi/val/*')
    os.system('rm -R data/slapi/test/*')    
    create_data_splits(test_video_code, train_ratio=0.6, val_ratio=0.4, split_file='data/slapi/SPLIT', include_file='data/slapi/INCLUDE', LABELED_FOLDER=LABELED_FOLDER, TRAIN_FOLDER=TRAIN_FOLDER, VAL_FOLDER=VAL_FOLDER, TEST_FOLDER=TEST_FOLDER)
    os.system(f'python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -av V3.{code}')
    

"""V4-V10: Model components
"""
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -tb -cc -rg l2 -av V4")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -cc -rg l2 -av V5")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -tb -rg l2 -av V6")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -rg l2 -av V7")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -tb -cc -av V8")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -cc -av V9")
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -tb -av V10")

"""V12
"""