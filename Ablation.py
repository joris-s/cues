import os
import random
import shutil
from pathlib import Path



import matplotlib
import matplotlib.pyplot as plt

# Set font parameters for Matplotlib
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

if os.name != 'nt':
    matplotlib.use('tkagg')

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

data = {
    'Accuracy':       [.15, .23, .25, .26, .33, .32, .39],
    'Precision':      [.15, .23, .24, .26, .30, .29, .30],
    'Recall':         [.22, .24, .32, .29, .33, .31, .35],
    'F1':             [.13, .18, .22, .23, .27, .26, .28]
}

def create_fig(data):
    markers = ['o', 's', 'D', 'p', 's']
    colors = ['#377eb8', '#ff7f00', '#984ea3', '#4daf4a', '#f781bf']  # These colors are a set of colorblind-friendly colors

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (key, values) in enumerate(data.items()):
        ax.plot(values, marker=markers[i], color=colors[i], label=key)

    ax.set_ylim(0, 0.5)
    plt.xlabel("Max. samples per class")
    plt.ylabel("Metric")

    # Set x-labels
    x_labels = ['1', '2', '3', '5', '10', '15', '25']
    plt.xticks(range(len(x_labels)), x_labels)

    # Move the legend outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Remove the upper and right boundaries
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('figs/abalation_v0.png', bbox_inches="tight", dpi=1000)
    #plt.show()

create_fig(data)

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
#os.system("python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -of -nt -av V2.tsne")

"""V3: Leave one out traing and testing
"""
def create_data_splits(test_video_code=None, train_ratio=0.6, val_ratio=0.2, split_file='data/slapi/SPLIT', include_file='data/slapi/INCLUDE'):
    # Read the INCLUDE file and create a set of video codes to be included
    with open(include_file, 'r') as f:
        included_video_codes = set(line.strip() for line in f.readlines())

    # Check if TRAIN_FOLDER, VAL_FOLDER, and TEST_FOLDER exist, create them if not
    for folder in [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Filter the video_codes based on the included_video_codes set
    video_codes = [vc for vc in os.listdir(LABELED_FOLDER) if vc in included_video_codes]

    # Create a dictionary of sets to keep track of classes per video code
    video_code_classes = {vc: set(os.listdir(os.path.join(LABELED_FOLDER, vc))) for vc in video_codes}

    # Find the intersection of all sets to get the classes that exist for all video codes
    common_classes = set.intersection(*video_code_classes.values())

    # Organize videos by class
    class_videos = {}
    for video_code in video_codes:
        video_code_path = os.path.join(LABELED_FOLDER, video_code)
        for class_name in common_classes:
            class_path = os.path.join(video_code_path, class_name)
            if class_name not in class_videos:
                class_videos[class_name] = []
            for video_file in os.listdir(class_path):
                class_videos[class_name].append((video_code, os.path.join(class_path, video_file)))

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
        for class_name, video_infos in class_videos.items():
            random.shuffle(video_infos)
            n = len(video_infos)
            train_count, val_count = max(1, int(n * train_ratio)), max(1, int(n * val_ratio))

            for idx, (video_code, video_path) in enumerate(video_infos):
                if video_code == test_video_code:  # If the video code is the test code
                    split = 'test'
                else:  # If the video code is not the test code
                    if idx < train_count:
                        split = 'train'
                    elif idx < train_count + val_count:
                        split = 'val'
                    else:  # This line is not necessary but is
                        continue  # Skip the remaining iterations if the video is neither for train nor val.

                # Write the split information to the text file
                f.write(f"{video_path} {split}\n")

                # Copy the video to the corresponding split folder
                dest_folder = os.path.join(split_data[split], class_name)
                Path(dest_folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, os.path.join(dest_folder, os.path.basename(video_path)))

#for i, code in enumerate(['417_20221012', '867_20221208', '702_20221115', '614_20221122']):
#    os.system('rm -R data/slapi/train/*')
#    os.system('rm -R data/slapi/val/*')
#    os.system('rm -R data/slapi/test/*')    
#    create_data_splits(test_video_code=code, train_ratio=0.6, val_ratio=0.4, split_file='data/slapi/SPLIT', include_file='data/slapi/INCLUDE')
#    os.system(f'python Main.py -b a2 -ep 5 -bs 4 -do 0.3 -cl 3 -sh 15 -av V3.{code}')
    

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
