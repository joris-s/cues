import os
import random
import shutil
from pathlib import Path
import json

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV

# Set font parameters for Matplotlib
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

if os.name != 'nt':
    matplotlib.use('tkagg')


"""*****************************************
*            Define Constants              *
*****************************************"""
OUTPUT_SIGNATURE = (
  tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
  tf.TensorSpec(shape = (), dtype = tf.int32),
  tf.TensorSpec(shape = (), dtype = tf.string)
)

GENERATOR_SIGNATURE = (
  tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), 
  tf.TensorSpec(shape = (), dtype = tf.int32), 
  tf.TensorSpec(shape = (), dtype = tf.int32)
)

STARTSTOP_SIGNATURE = (
  #tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), 
  tf.TensorSpec(shape = (), dtype = tf.int32), 
  tf.TensorSpec(shape = (), dtype = tf.int32)
)

MOVINET_PARAMS = {
  'a0': (172, 5),
  'a1': (172, 5),
  'a2': (224, 5),
  'a3': (256, 12),
  'a4': (290, 8),
  'a5': (320, 12)
}

FPS = 30
META_CLASSES = 101


LABELED_FOLDER = 'data/slapi/labeled'
UNLABELED_FOLDER = 'data/self/long_med'
UNLABELED_PATH = 'data/self/long.mp4'
TRAIN_FOLDER = 'data/self/joris'
VAL_FOLDER = 'data/self/ercan'
TEST_FOLDER = 'data/self/roos'
META_TRAIN_FOLDER = 'data/UCF-101/train'
META_VAL_FOLDER = 'data/UCF-101/val'
AL_FOLDER = 'data/slapi/active-learning'
if (os.name == 'nt') == False:
    LABELED_FOLDER = 'data/slapi/labeled'
    UNLABELED_FOLDER = 'data/slapi/unlabeled'
    UNLABELED_PATH = 'data/slapi/unlabeled/cropped_top_657_20230321.mp4'
    TRAIN_FOLDER = 'data/slapi/train'
    VAL_FOLDER = 'data/slapi/val'
    TEST_FOLDER = 'data/slapi/test'
    META_TRAIN_FOLDER = 'data/UCF-101/train'
    META_VAL_FOLDER = 'data/UCF-101/val'
    AL_FOLDER = 'data/slapi/labeled/active-learning'

LABEL_NAMES = sorted(os.listdir(TRAIN_FOLDER))

"""*****************************************
*                Metrics                   *
*****************************************"""

class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_true_one_hot = tf.squeeze(y_true_one_hot, axis=-2)
        return super().update_state(y_true_one_hot, y_pred, sample_weight=sample_weight)

class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_true_one_hot = tf.squeeze(y_true_one_hot, axis=-2)
        return super().update_state(y_true_one_hot, y_pred, sample_weight=sample_weight)
            
class SparseF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='sparse_f1_score', **kwargs):
        super(SparseF1Score, self).__init__(name=name, **kwargs)
        self.precision = SparsePrecision()
        self.recall = SparseRecall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1_score

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()

METRICS = [
    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    SparsePrecision(name='precision'),
    SparseRecall(name='recall'),
    SparseF1Score(name='f1')
]


"""*****************************************
*      Frame Generator Functions           *
*****************************************"""
def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size, frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))

    if not src.isOpened():
        print(f"Error: Could not open the video file {video_path}.")
        return result

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(video_length / n_frames)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()

    for _ in range(n_frames):
        for _ in range(frame_step-1):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros((*output_size, 3)))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

def frames_from_video_of(video_path, n_frames, output_size, frame_step=15):
    optical_flow_frames = []
    src = cv2.VideoCapture(str(video_path))

    if not src.isOpened():
        print(f"Error: Could not open the video file {video_path}.")
        return optical_flow_frames

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(video_length / n_frames)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for _ in range(n_frames):
        for _ in range(frame_step-1):
            ret, frame = src.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Using flow frames directly and appending gray frame to make it 3D
            optical_flow_frame = np.dstack((flow[..., 0], flow[..., 1], gray))

            # Format frame to required output size
            optical_flow_frame = format_frames(optical_flow_frame, output_size)
            optical_flow_frames.append(optical_flow_frame)
            prev_gray = gray
        else:
            optical_flow_frames.append(np.zeros((*output_size, 3)))
            
    src.release()
    optical_flow_frames = np.array(optical_flow_frames)

    return optical_flow_frames

def filter_examples_per_class(examples_list, max_examples_per_class):
    num_examples_per_class = {}
    filtered_list = []
    random.shuffle(examples_list)
    for example in examples_list:
        if num_examples_per_class.setdefault(example[1], 0) < max_examples_per_class:
            num_examples_per_class[example[1]] += 1
            filtered_list.append(example)
    return filtered_list

class FrameGenerator:
    def __init__(self, path, n_frames, resolution, training=False, extension='.mp4', shots=-1, frame_step=15):
        self.path = Path(path)
        self.n_frames = n_frames
        self.resolution = resolution
        self.training = training
        self.extension = extension
        self.shots = shots
        self.frame_step = frame_step
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*' + self.extension))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.shots > -1:
            pairs = filter_examples_per_class(pairs, self.shots)

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames, output_size=(self.resolution, self.resolution), frame_step=self.frame_step)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, tf.cast(label, tf.int32), path.__str__()

class ProposalGenerator:
    def __init__(self, path, indices, n_frames, resolution, frame_step, extension='.mp4'):
        self.path = path
        self.indices = indices
        self.n_frames = n_frames
        self.resolution = resolution
        self.extension = extension
        self.frame_step = frame_step
        
    def generate_frames(self, src, n_frames, output_size, start, stop, frame_step=15):
        result = []
    
        if not src.isOpened():
            print(f"Error: Could not open the video file.")
            return result
    
        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = src.read()
        
        video_length = stop-start
        frame_step = int(video_length / n_frames)
    
        for _ in range(n_frames):
            for _ in range(frame_step-1):
                ret, frame = src.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros((*output_size, 3)))

        result = np.array(result)[..., [2, 1, 0]]
        
        return result
    

    def __call__(self):
        src = cv2.VideoCapture(str(self.path))
        for (start_index, stop_index) in self.indices:
            video_frames = self.generate_frames(src, self.n_frames, output_size=(self.resolution, self.resolution), 
                                                  start=start_index, stop=stop_index, frame_step=self.frame_step)
            yield video_frames, tf.cast(start_index, tf.int32), tf.cast(stop_index, tf.int32)
        src.release()

class StartStopGenerator:
    def __init__(self, model, path, n_frames, resolution, frame_step, extension='.mp4', start=0):
        self.model = model
        self.path = path
        self.start = start
        self.n_frames = n_frames
        self.output_size = (resolution, resolution)
        self.extension = extension
        self.frame_step = frame_step

    def process_frames(self, src):
        frames = []
        try:
            for _ in range(self.n_frames):
                for _ in range(self.frame_step-1):
                    ret, frame = src.read()
                if ret:
                    frame = format_frames(frame, self.output_size)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((*self.output_size, 3)))
        except Exception as e:
            print(f'Error occured, stopping instance generation: {e}')
            return False
        return np.array(frames)[..., [2, 1, 0]]
    
    def get_label(self, result):
        return np.argmax(self.model.predict(result[np.newaxis, :])[0])

    def sliding_frames_from_video(self, starting_frame, src, max_combined=3):
        src.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        result = self.process_frames(src)
        next_result = self.process_frames(src)
                
        if isinstance(result, bool):
            return False, None, None
        if isinstance(next_result, bool):
            return False, None, None
        
        label, next_label = self.get_label(result), self.get_label(next_result)
        counter = 0

        while label == next_label and counter < max_combined:
            result = np.vstack((result, next_result.copy()))
            result = result[::2]
            next_result = self.process_frames(src)
            if isinstance(next_result, bool):
                return False, None, None
            label = next_label 
            next_label = self.get_label(next_result)
            counter += 1

        stop_index = src.get(cv2.CAP_PROP_POS_FRAMES)
        return result, int(starting_frame), int(stop_index)

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.start = np.random.randint(0, 1000) #10 min
        starting_frame = self.start

        while starting_frame < (self.start+18000):#:(total_frames - self.n_frames*self.frame_step*2):
            print(starting_frame)

            processed_frames, start_index, stop_index = self.sliding_frames_from_video(starting_frame, cap)
            if isinstance(processed_frames, np.ndarray) == False:
                break
            
            starting_frame = stop_index + 1
            yield tf.cast(start_index, tf.int32), tf.cast(stop_index, tf.int32)


"""*****************************************
*             Helper Functions             *
*****************************************"""
def get_actual_predicted_labels(dataset, model):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)
    
    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)
    
    return actual, predicted

def remove_paths(ds):
    def select_frames_and_labels(frames, labels, paths):
        return frames, labels
    return ds.map(select_frames_and_labels)

def remove_indices(ds):
    def select_frames(frames, start, stop):
        return frames
    return ds.map(select_frames)

def scale_class_weights(class_weights, target_loss, current_loss, num_classes):
    loss_ratio = target_loss / current_loss
    scaling_factor = loss_ratio / num_classes
    scaled_weights = {k: v * scaling_factor for k, v in class_weights.items()}
    return scaled_weights

def get_class_weights(ds):
    labels = [int(label) for _, label in ds.unbatch()]
    total_labels = len(labels)
    class_weights = {i: 1/(labels.count(i)/total_labels) for i in range(len(list(set(labels))))}
    class_weight_sum = sum(class_weights.values())
    class_weights = {k: v/class_weight_sum for k, v in class_weights.items()}
    return class_weights

def create_data_splits(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, split_file='data/slapi/SPLIT', include_file='data/slapi/INCLUDE'):
    # Read the INCLUDE file and create a set of video codes to be included
    with open(include_file, 'r') as f:
        included_video_codes = set(line.strip() for line in f.readlines())
        
    # Check if TRAIN_FOLDER, VAL_FOLDER, and TEST_FOLDER exist, create them if not
    for folder in [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Filter the video_codes based on the included_video_codes set
    video_codes = [vc for vc in os.listdir(LABELED_FOLDER) if vc in included_video_codes]
    
    # Organize videos by class
    class_videos = {}
    for video_code in video_codes:
        video_code_path = os.path.join(LABELED_FOLDER, video_code)
        for class_name in os.listdir(video_code_path):
            if class_name == 'ALL':  # Skip the 'ALL' folder
                continue
            
            class_path = os.path.join(video_code_path, class_name)
            if class_name not in class_videos:
                class_videos[class_name] = []
            for video_file in os.listdir(class_path):
                class_videos[class_name].append(os.path.join(class_path, video_file))

    # Remove classes with fewer than 3 samples
    class_videos = {k: v for k, v in class_videos.items() if len(v) >= 3}

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
        for class_name, videos in class_videos.items():
            random.shuffle(videos)
            n = len(videos)
            train_count, val_count = max(1, int(n * train_ratio)), max(1, int(n * val_ratio))
            
            for idx, video_path in enumerate(videos):
                if idx < train_count:
                    split = 'train'
                elif idx < train_count + val_count:
                    split = 'val'
                else:
                    split = 'test'
                
                # Write the split information to the text file
                f.write(f"{video_path} {split}\n")

                # Copy the video to the corresponding split folder
                dest_folder = os.path.join(split_data[split], class_name)
                Path(dest_folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, os.path.join(dest_folder, os.path.basename(video_path)))



"""*****************************************
*             Plotting Functions           *
*****************************************"""
def cm_heatmap_small(actual, predicted, labels, savefigs=False, name='heatmap'):
    plt.clf()
    mapping = {
        0:  2,
        1:  2,
        2:  0,
        3:  0,
        4:  1,
        5:  1,
        6:  1,
        7:  1,
        8:  0,
        9:  2,
        10: 2,
        11: 0,
        12: 0,
        13: 0,
        14: 2,
        15: 0,
        16: 0,
        17: 0,
        18: 2,
        19: 2,
        20: 1,
        21: 0,
        22: 2,
        23: 2
    }
    
    # First, map classes to categories
    actual = [mapping[int(x)] for x in actual]
    predicted = [mapping[int(x)] for x in predicted]

    # Get the unique category labels
    labels = list(set(mapping.values()))

    # Compute confusion matrix
    cm_num = confusion_matrix(actual, predicted)
    cm = []
    for i in range(len(cm_num)):
        row = cm_num[i]
        cm.append([(round(x/sum(row), 2)) for x in row])

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the normalized confusion matrix
    heatmap1 = sns.heatmap(cm, annot=True, cbar=False, cmap='BuPu', vmin=0.00, vmax=1.00, ax=ax1, square=True)
    ax1.set_title('Normalized Confusion Matrix', fontsize=16)
    ax1.set_xlabel('Predicted Category', fontsize=16)
    ax1.set_ylabel('Actual Category', fontsize=16)
    ax1.xaxis.tick_bottom()
    plt.setp(ax1.get_xticklabels(), rotation=90)
    plt.setp(ax1.get_yticklabels(), rotation=0)
    ax1.xaxis.set_ticklabels(['Hunger', 'Discomfort', 'Other'])
    ax1.yaxis.set_ticklabels(['Hunger', 'Discomfort', 'Other']) 
    
    # Add black borders around the diagonal
    for i in range(len(labels)):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax1.add_patch(rect)

    # Plot the original confusion matrix with integer values
    #heatmap2 = sns.heatmap(cm_num, annot=True, cbar=False, cmap=ListedColormap(['white']), fmt='d', ax=ax2)
    heatmap2 = sns.heatmap(cm, annot=True, cbar=False, cmap='BuPu', vmin=0.00, vmax=1.00, ax=ax2, square=True)
    ax2.set_title('Confusion Matrix with Frequency Values', fontsize=16)
    ax2.set_xlabel('Predicted Category', fontsize=16)
    ax2.set_ylabel('Actual Category', fontsize=16)
    ax2.xaxis.tick_bottom()
    plt.setp(ax2.get_xticklabels(), rotation=90)
    plt.setp(ax2.get_yticklabels(), rotation=0)
    ax2.xaxis.set_ticklabels(['Hunger', 'Discomfort', 'Other'])
    ax2.yaxis.set_ticklabels(['Hunger', 'Discomfort', 'Other']) 
    
    # Customize annotations to make non-zero values bold
    counter = 0
    for text1, text2 in zip(heatmap1.texts, heatmap2.texts):
        text2.set_text(str(np.array(cm_num).flatten()[counter]))
        counter+=1
        if text1.get_text() != "0":
            text1.set_weight("bold")
            text2.set_weight("bold")
        if text1.get_text() == "0":
            text1.set_text("-")
            text2.set_text("-")

    # Add black borders around the diagonal
    for i in range(len(labels)):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax2.add_patch(rect)
        
    plt.tight_layout()
    
    if savefigs:
        plt.savefig('figs/cm/'+name+'_small.png', bbox_inches='tight', dpi=1000)
    plt.close()    
    plt.clf()
    
def cm_heatmap(actual, predicted, labels, savefigs=False, name='heatmap'):
    plt.clf()
    cm_num = confusion_matrix(actual, predicted)
    cm = []
    for i in range(len(cm_num)):
        row = cm_num[i]
        cm.append([(round(x/sum(row), 2)) for x in row])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 30))
    
    # Plot the normalized confusion matrix
    heatmap1 = sns.heatmap(cm, annot=True, cbar=False, cmap='BuPu', vmin=0.00, vmax=1.00, ax=ax1, square=True)
    ax1.set_title('Normalized Confusion Matrix', fontsize=20)
    ax1.set_xlabel('Predicted Action', fontsize=16)
    ax1.set_ylabel('Actual Action', fontsize=16)
    ax1.xaxis.tick_bottom()
    plt.setp(ax1.get_xticklabels(), rotation=90)
    plt.setp(ax1.get_yticklabels(), rotation=0)
    ax1.xaxis.set_ticklabels(labels)
    ax1.yaxis.set_ticklabels(labels)
    
    # Add black borders around the diagonal
    for i in range(len(labels)):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax1.add_patch(rect)

    # Plot the original confusion matrix with integer values
    #heatmap2 = sns.heatmap(cm_num, annot=True, cbar=False, cmap=ListedColormap(['white']), fmt='d', ax=ax2)
    heatmap2 = sns.heatmap(cm, annot=True, cbar=False, cmap='BuPu', vmin=0.00, vmax=1.00, ax=ax2, square=True)
    ax2.set_title('Confusion Matrix with Frequency Values', fontsize=20)
    ax2.set_xlabel('Predicted Action', fontsize=16)
    ax2.set_ylabel('Actual Action', fontsize=16)
    ax2.xaxis.tick_bottom()
    plt.setp(ax2.get_xticklabels(), rotation=90)
    plt.setp(ax2.get_yticklabels(), rotation=0)
    ax2.xaxis.set_ticklabels(labels)
    ax2.yaxis.set_ticklabels(labels)
    
    # Customize annotations to make non-zero values bold
    counter = 0
    for text1, text2 in zip(heatmap1.texts, heatmap2.texts):
        text2.set_text(str(np.array(cm_num).flatten()[counter]))
        counter+=1
        if text1.get_text() != "0":
            text1.set_weight("bold")
            text2.set_weight("bold")
        if text1.get_text() == "0":
            text1.set_text("-")
            text2.set_text("-")

    # Add black borders around the diagonal
    for i in range(len(labels)):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax2.add_patch(rect)
        
    plt.tight_layout()
    
    if savefigs:
        plt.savefig('figs/cm/'+name+'.png', bbox_inches='tight', dpi=1000)
    plt.close()    
    plt.clf()
    
    cm_heatmap_small(actual, predicted, ['Hunger', 'Discomfort', 'Other'], savefigs, name)

# Modified plot_train_val function
def plot_metrics(history, metrics, title, savefigs=True):
    plt.clf()
    num_metrics = len(metrics)
    num_rows = np.ceil(num_metrics / 2).astype(int)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))
    
    for i, metric in enumerate(metrics):
        row, col = divmod(i, 2)
        
        if metric == 'loss':
            train_metric_key = metric
            val_metric_key = f"val_{metric}"
        else:
            train_metric_key = f"train_{metric}"
            val_metric_key = f"val_{metric}"
            
        
        train_metric = history[train_metric_key]
        val_metric = history[val_metric_key]
        
        axes[row, col].plot(train_metric, label='Train', marker='o')
        axes[row, col].plot(val_metric, label='Validation', marker='^')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(metric.upper(), fontsize=16, fontdict=dict(weight='bold'))
        axes[row, col].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].spines['right'].set_visible(False)

        if metric != 'loss' and not metric.startswith('train_loss'):
            axes[row, col].set_ylim([0, 1.1])

        axes[row, col].legend(loc='best')

    if num_metrics % 2 != 0:
        fig.delaxes(axes[-1, -1])
        
    #fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if savefigs:
        plt.savefig(f'figs/metrics/{title}.png', dpi=1000)
    plt.close()
    plt.clf()

def plot_metrics_from_file(file_path, title, savefigs=True):
    with open(file_path, 'r') as f:
        history = json.load(f)

    metrics = ['loss', 'accuracy', 'balanced accuracy', 'precision', 'recall', 'f1']
    plot_metrics(history, metrics, title, savefigs)
    
def batch_processing(vids, batch_size):
      num_batches = int(np.ceil(len(vids) / batch_size))
      for i in range(num_batches):
          start = i * batch_size
          end = min((i+1) * batch_size, len(vids))
          yield vids[start:end]

def count_elements_in_dataset(dataset):
    count = 0
    for _ in dataset.unbatch():
        count += 1
    return count

def plot_tsne(tsne_representation, labels, indices, savefigs=True, name='', x_lim=None, y_lim=None):
    num_classes = len(np.unique(labels))
    colors = plt.cm.tab20b(np.linspace(0, 1, num_classes))
    markers = ['s', '>', 'D', 'X', 'v', 'H', 
               '^', '2', '_', '1', 'x', 'p', 
               '+', 'o', 'h', 'P', '3', '*', 
               'd', ',', '4', '.', '<', '|']
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for label, color, marker in zip(unique_labels, colors, markers[:num_classes]):
        label_indices = np.where(labels == label)
        plot_indices = np.intersect1d(label_indices, indices)
        plt.scatter(tsne_representation[plot_indices, 0], tsne_representation[plot_indices, 1], label=LABEL_NAMES[label], c=[color], marker=marker, alpha=0.7, s=150)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel("t-SNE Axis 0")
    plt.ylabel("t-SNE Axis 1")
    plt.legend(title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
    if not x_lim:
        x_lim = (min(tsne_representation[:, 0]) - 5, max(tsne_representation[:, 0]) + 5)
        y_lim = (min(tsne_representation[:, 1]) - 5, max(tsne_representation[:, 1]) + 5)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    print('saving')
    if not os.path.exists('figs/'):
        os.makedirs('figs/')

    plt.savefig(os.path.join('figs/tsne/', f'{name}.png'), dpi=1000, bbox_inches='tight')
    plt.close()
    
    return x_lim, y_lim

def get_video_codes(video_paths):
    # Initialize a dictionary to hold the mapping
    video_mapping = {}

    # Get all the video paths in the 'data/slapi/labeled' directory
    for root, dirs, files in os.walk('data/slapi/labeled'):
        for file in files:
            if file.endswith('.mp4'):
                video = str(root).split('/')[3]
                video_mapping[file] = video

    # Initialize a list to hold the video codes
    video_codes = []

    # For each video path in the provided list, find the VIDEO
    for video_path in video_paths:
        video_path = video_path.numpy().decode('utf-8')
        file = video_path.split('/')[-1]
        video = video_mapping.get(file, "Not found")
        video_codes.append(video[:2])

    return np.array(video_codes)

def plot_tsne_per_video(tsne_representation, paths, indices, savefigs=True, name='', x_lim=None, y_lime=None):
    labels = get_video_codes(paths)
    
    num_classes = len(np.unique(labels))
    markers = ['o', 's', '*', 'D', 'p', 's']
    colors = ['#377eb8', '#ff7f00', '#a65628', '#984ea3', '#4daf4a', '#f781bf']  # These colors are a set of colorblind-friendly colors

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)

    for label, color, marker in zip(unique_labels, colors, markers[:num_classes]):
        label_indices = np.where(labels == label)
        plt.scatter(tsne_representation[label_indices, 0], tsne_representation[label_indices, 1], label=label, c=[color], marker=marker, alpha=0.7, s=150)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel("t-SNE Axis 0")
    plt.ylabel("t-SNE Axis 1")
    plt.legend(title="Infant ID", loc='center left', bbox_to_anchor=(1, 0.5))
    if not x_lim:
        x_lim = (min(tsne_representation[:, 0]) - 5, max(tsne_representation[:, 0]) + 5)
        y_lim = (min(tsne_representation[:, 1]) - 5, max(tsne_representation[:, 1]) + 5)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    print('saving')
    if not os.path.exists('figs/'):
        os.makedirs('figs/')

    plt.savefig(os.path.join('figs/tsne/', f'{name}.png'), dpi=1000, bbox_inches='tight')
    plt.close()
    
    return x_lim, y_lim   

def plot_all_tsne(model):
    combined_datasets = model.train_ds.concatenate(model.val_ds).concatenate(model.test_ds)

    data = [(v, start, stop) for (v, start, stop) in combined_datasets.unbatch()]
    vids = np.array([v for v, _, _ in data])
    labels = np.array([l for _, l, _ in data])
    paths = np.array([p for _, _, p in data])

    # Get the penultimate layer of the model in batches
    batch_size = 16
    penultimate_features = []
    for i, batch_vids in enumerate(batch_processing(vids, batch_size)):
        print(f"Processing batch {i + 1}")
        features = model.base_model.backbone(batch_vids)
        batch_features = [f.numpy().flatten() for f in features[0]['head']]
        penultimate_features.extend(batch_features)
    penultimate_features = np.array(penultimate_features)
    print("Penultimate features obtained")

    # Compute t-SNE representation for all the data at once
    tsne = TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(penultimate_features)
    print("t-SNE representation computed")

    # Compute lengths of the train, val, and test datasets
    train_len = count_elements_in_dataset(model.train_ds)
    val_len = count_elements_in_dataset(model.val_ds)
    test_len = count_elements_in_dataset(model.test_ds)
    print(f"Lengths - Train: {train_len}, Val: {val_len}, Test: {test_len}")

    # Plot t-SNE for train, val, test, and combined datasets
    x_lim, y_lim = plot_tsne_per_video(tsne_representation, paths, np.arange(len(labels)), name=f'tsne_{model.model_id.upper()}_video_combined')
    x_lim, y_lim = plot_tsne(tsne_representation, labels, np.arange(len(labels)), name=f'tsne_{model.model_id.upper()}_combined')
    plot_tsne(tsne_representation[:train_len], labels[:train_len], np.arange(train_len), name=f'tsne_{model.model_id.upper()}_train', x_lim=x_lim, y_lim=y_lim)
    plot_tsne(tsne_representation[train_len:train_len+val_len], labels[train_len:train_len+val_len], np.arange(val_len), name=f'tsne_{model.model_id.upper()}_val', x_lim=x_lim, y_lim=y_lim)
    plot_tsne(tsne_representation[train_len+val_len:], labels[train_len+val_len:], np.arange(test_len), name=f'tsne_{model.model_id.upper()}_test', x_lim=x_lim, y_lim=y_lim)


"""*****************************************
*             MoViNet Helpers              *
*****************************************"""
    
def AIPCreateBackboneAndClassifierModel(model_id, 
                                        num_classes, 
                                        frames_number, 
                                        batch_size, 
                                        resolution, 
                                        train_whole_model, dropout,
                                        checkpoint_dir,
                                        causal_conv=False,
                                        conv_type: str = '3d', 
                                        se_type: str = '3d', 
                                        activation: str = 'swish',
                                        gating_activation: str = 'sigmoid', 
                                        stream_mode=False, 
                                        load_pretrained_weights=True, 
                                        regularization=None):
 
  tf.keras.backend.clear_session()
  backbone = movinet.Movinet(model_id=model_id,
                            causal=causal_conv,
                            conv_type=conv_type,
                            se_type=se_type,
                            activation=activation,
                            gating_activation=gating_activation,
                            use_external_states=stream_mode
                            )

  backbone.trainable = train_whole_model
  
  model = None
  if load_pretrained_weights:
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([None, None, None, None, 3])
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()  
 
  model = movinet_model.MovinetClassifier(
        backbone=backbone, 
        activation=activation,
        num_classes=num_classes, 
        output_states=stream_mode,
        dropout_rate = dropout,
        kernel_regularizer=regularization
        )
  
  model.build([batch_size, frames_number, resolution, resolution, 3])
  return model

def predict_video(path, model, n_frames = 24, frame_step = 6):
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open the video file")
        
    ret, frame = cap.read()
    predictions = []
    while ret:
        
        frames = []
        for i in range(model.batch_size*2):
            result = []
            for _ in range(n_frames):
                for _ in range(frame_step):
                    ret, frame = cap.read()
                    
                if not ret:
                    result.append(np.zeros((model.resolution, model.resolution, 3)))
                else:
                    frame = format_frames(frame, (model.resolution, model.resolution))
                    result.append(frame)
                    
            result = np.array(result)[..., [2, 1, 0]]
            frames.append(result)
    
        predicted = model.base_model.predict(np.array(frames))
        predicted = tf.concat(predicted, axis=0)
        predicted = list(tf.argmax(predicted, axis=1).numpy())
        
        predictions.extend(predicted)
        
    # Convert predictions to one-hot-encoded format
    num_classes = len(LABEL_NAMES)
    one_hot_predictions = np.zeros((num_classes, len(predictions)))
    for i, pred in enumerate(predictions):
        one_hot_predictions[pred, i] = 1

    # Calculate the time_points considering each prediction is made every 4.8 seconds
    duration = (n_frames*frame_step)/30
    time_points = np.arange(0, len(predictions)*duration/60, duration/60)  # convert seconds to minutes

    # Create the heatmap
    plt.figure(figsize=(30, 10))
    plt.imshow(one_hot_predictions, aspect='auto', cmap='binary', origin='lower')

    # Label the y-axis with class numbers
    plt.yticks(np.arange(num_classes), LABEL_NAMES)

    # Define the xticks to be every 5 minutes. Since each tick is 4.8 seconds apart, we take 5*60/4.8 = 62.5 ticks to be equivalent to 5 minutes.
    # Use dtype=int to ensure that the labels are integers
    plt.xticks(np.arange(0, len(time_points), int(5*60/4.8)), np.arange(0, len(time_points)*4.8/60, 5, dtype=int))

    plt.xlabel('Time (minutes)')
    plt.ylabel('Predicted Class')
    
    # Hide the top and right box lines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig(os.path.join('figs/', 'video_prediction.png'), dpi=1000, bbox_inches='tight')        

    return predictions

def plot_confusion_matrices(matrices, name='cm_probabilities'):
    plt.clf()
    fig, axn = plt.subplots(1, 4, sharex=True, sharey=False, figsize=(30, 7.5))
    cbar_ax = fig.add_axes([.91, .12, .03, .78])  # Positioning the colorbar
    labels = ['Baseline', 'Iteration 1', 'Iteration 2', 'Iteration 3']

    for i, ax in enumerate(axn.flat):
        sns.heatmap(matrices[i,:,:], ax=ax,
                    cbar=i == 0, square=True,
                    vmin=0, vmax=0.5,  # Assuming your probabilities are between 0 and 1
                    cbar_ax=None if i else cbar_ax, 
                    cmap='hot')
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks(np.arange(24)+0.5, LABEL_NAMES)

        plt.setp(ax.get_yticklabels(), rotation=0)
        
        if i==0: 
            ax.set_ylabel('Samples')
        else:
            ax.set_yticks([])
            
        ax.set_xlabel(labels[i])  # Set label (used for title) on x-axis

    plt.savefig(os.path.join('figs/', f'{name}.png'), dpi=1000, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    predictionsV12 = np.array([
    [[0.03396966,0.09698446,0.0114843,0.02584215,0.2193871,0.01015038,0.07310229,0.03349538,0.06082897,0.07071212,0.02887399,0.02025519,0.04221416,0.03362594,0.02594332,0.01797654,0.0142306,0.07253677,0.00954524,0.01452277,0.0111617,0.01036505,0.04836014,0.01443177],[0.22011848,0.0308996,0.04035672,0.02725674,0.01447714,0.01310599,0.01638908,0.03097068,0.01043737,0.13082132,0.01454384,0.02206193,0.07284816,0.09193093,0.02483503,0.00986956,0.0280393,0.01418165,0.00903637,0.03916787,0.01603791,0.0116704,0.10317732,0.00776661],[0.01656928,0.00412642,0.00466346,0.02808302,0.00427996,0.06055374,0.00700734,0.03504635,0.02370742,0.0065032,0.10444608,0.00464587,0.01223274,0.01077731,0.02513094,0.03394099,0.11626545,0.00283987,0.11042086,0.04383835,0.2448161,0.05248374,0.00386624,0.04375524],[0.03940048,0.00667163,0.00189736,0.00333978,0.00765076,0.13245566,0.01370066,0.08749659,0.05716874,0.00755833,0.1543595,0.00366487,0.01098621,0.00814554,0.00673306,0.01895455,0.01476046,0.00402216,0.04948225,0.01056419,0.08616022,0.05554565,0.00441943,0.21486185],[0.19618145,0.05102706,0.09132596,0.09069141,0.07323393,0.01137662,0.0309818,0.02371597,0.02461988,0.03924051,0.02735765,0.01157736,0.04203419,0.03444465,0.02957716,0.01831597,0.02821237,0.03313388,0.01827277,0.02939011,0.03167548,0.01260211,0.04339637,0.00761539],[0.01235774,0.05503628,0.02155036,0.01497454,0.05020824,0.03977954,0.17342402,0.07768513,0.07587195,0.02476272,0.03624792,0.0532066,0.01960888,0.02468341,0.03879626,0.02433861,0.0134163,0.05460035,0.05716778,0.02382135,0.01882401,0.02682235,0.02662212,0.03619342],[0.01743672,0.03881676,0.06649295,0.02212557,0.0157772,0.04203983,0.03360539,0.01937231,0.02102436,0.03057925,0.02589145,0.18929812,0.02793659,0.08807912,0.07550994,0.0440923,0.03188462,0.02798664,0.03028113,0.04758925,0.01967596,0.04576018,0.02398936,0.01475495],[0.01693899,0.066234,0.01497053,0.01566146,0.04091243,0.06011748,0.06121111,0.03020181,0.0624446,0.02608054,0.03967323,0.09486283,0.02416155,0.05390059,0.05440589,0.0764915,0.01934904,0.05044024,0.03131266,0.01774964,0.02139947,0.0604071,0.02678912,0.03428422],[0.01719372,0.01456248,0.00721728,0.02146364,0.00706611,0.05452206,0.01900577,0.02966752,0.01422674,0.0829023,0.02350667,0.03709618,0.09091507,0.07234903,0.06422733,0.02599408,0.15599813,0.01461409,0.02475869,0.09130532,0.01600325,0.04384674,0.05318033,0.01837744],[0.09890362,0.04908106,0.01647054,0.01251672,0.05151329,0.0106508,0.0218433,0.04579416,0.02213348,0.19133952,0.01464766,0.01981571,0.11276402,0.07916363,0.0330968,0.00654056,0.03222108,0.02697052,0.00696002,0.03385214,0.01109959,0.00755523,0.08740132,0.00766513],[0.02720785,0.02316108,0.03390696,0.02254269,0.00924861,0.04807391,0.02006422,0.02162034,0.01578902,0.02327287,0.03061977,0.07855939,0.02608525,0.0736409,0.12367241,0.06448443,0.07329161,0.01935412,0.03103717,0.133565,0.02130387,0.0505585,0.01773591,0.01120419],[0.02979165,0.0654831,0.01810943,0.01048883,0.04444187,0.06068251,0.05045732,0.03393144,0.06288064,0.02803476,0.03030489,0.09327928,0.02840576,0.05940249,0.05883723,0.06455557,0.01649699,0.05439194,0.02416283,0.01157951,0.01993126,0.06341151,0.03921731,0.03172182],[0.01700081,0.0072067,0.00717531,0.03127242,0.00625172,0.06423521,0.01279929,0.04764756,0.02882626,0.01247525,0.0797089,0.00584029,0.01045469,0.01056304,0.01921975,0.03024458,0.04314202,0.00461697,0.20136085,0.05880903,0.1725744,0.06041009,0.00743392,0.0607309,],[0.02183042,0.07666985,0.01520349,0.01084814,0.05987455,0.0491553,0.06576572,0.03592722,0.07861306,0.02434816,0.03689732,0.08812683,0.02404667,0.05135041,0.04998068,0.06547381,0.01338683,0.05528592,0.02776315,0.01168056,0.02327043,0.04577791,0.02872797,0.03999558],[0.01391146,0.04458304,0.02940755,0.02866533,0.05385989,0.02391247,0.12031627,0.02213225,0.04279818,0.03215986,0.01733249,0.06895069,0.04313164,0.04686208,0.04246556,0.02186069,0.01737097,0.14382872,0.02060691,0.01726227,0.00993424,0.03069548,0.08617936,0.02177264],[0.05142184,0.03763122,0.00955165,0.00955375,0.02974095,0.04138439,0.01607587,0.03972946,0.02846031,0.21141799,0.02319778,0.02920022,0.1001775,0.08926006,0.02671567,0.01923663,0.03784289,0.02259557,0.00897812,0.02520178,0.01219107,0.02362945,0.08848968,0.01831621],[0.00809498,0.00845869,0.01979614,0.02396507,0.0046346,0.12868714,0.01711654,0.03065863,0.03257547,0.00995795,0.05379942,0.02039416,0.00788787,0.01154357,0.03598832,0.05809745,0.04584339,0.00637532,0.10897707,0.05768433,0.07644883,0.12101909,0.00516874,0.10682726],[0.02719516,0.03146231,0.03155925,0.02895318,0.00953272,0.0213724,0.0176397,0.01718463,0.01017778,0.23816085,0.01552826,0.02772161,0.1110827,0.10987573,0.03632078,0.01117257,0.07027148,0.0138281,0.00949818,0.06425544,0.01661729,0.01842334,0.05530232,0.00686434],[0.04665164,0.01646276,0.00585772,0.02110123,0.03682097,0.069327,0.01571933,0.05792584,0.08310422,0.01709186,0.1202306,0.00646954,0.01990014,0.01572478,0.022479,0.03989789,0.05079071,0.00974989,0.05626406,0.03588369,0.0982108,0.05961867,0.01187766,0.08283991],[0.02476264,0.15963484,0.01752706,0.00917751,0.21534222,0.02273843,0.06358486,0.0296953,0.09293149,0.02181267,0.02899332,0.03599504,0.01361863,0.02365578,0.03221731,0.03986201,0.00549649,0.06654105,0.01325767,0.00602412,0.01243891,0.01939015,0.02191434,0.02338818],[0.02459036,0.03546333,0.01978752,0.01754271,0.01735421,0.01921396,0.03539698,0.01667415,0.01390494,0.12639612,0.01219409,0.07153618,0.0919356,0.11708982,0.03524373,0.01491703,0.03085015,0.04067245,0.00869463,0.02603458,0.00927142,0.02369987,0.18241702,0.00911913],[0.03338659,0.10388594,0.01883627,0.01201976,0.08875149,0.04046208,0.0421046,0.02481388,0.06886493,0.04038595,0.02524127,0.07344362,0.02714232,0.0529043,0.0672982,0.06015933,0.01253524,0.07110941,0.01712014,0.01241671,0.0125317,0.03558961,0.03916898,0.01982765],[0.00242909,0.00343403,0.05291818,0.54384124,0.00173408,0.01183919,0.00659657,0.00524949,0.00385427,0.00738582,0.01567003,0.00595798,0.00453085,0.00701352,0.01874783,0.01602735,0.06618347,0.00278007,0.04413434,0.12616661,0.02873676,0.01695983,0.00415072,0.0036587,],[0.01800853,0.00386777,0.02682617,0.05901866,0.00156871,0.0379281,0.00455794,0.01750372,0.00723349,0.01688658,0.02618814,0.00605213,0.0143911,0.01425238,0.03411171,0.03122747,0.17163484,0.00195897,0.06283455,0.11903565,0.26115566,0.04296457,0.01150227,0.00929092]],[[5.20443823e-03,5.74616604e-02,4.40403819e-03,5.75245544e-02,4.15999830e-01,1.01386767e-03,5.54352030e-02,2.42290664e-02,3.22733819e-02,4.71969023e-02,1.25530669e-02,6.73918007e-03,4.53512929e-02,5.86658381e-02,1.12517029e-02,4.76873573e-03,6.84701139e-03,1.29779547e-01,1.44002726e-03,3.13256541e-03,1.46535225e-03,1.40179775e-03,1.31198345e-02,2.74099270e-03],[1.90458015e-01,1.10217100e-02,2.02318266e-01,1.77214846e-01,7.02571683e-03,5.82417939e-03,4.52496624e-03,2.30327360e-02,1.33425090e-02,8.08499083e-02,8.64979811e-03,1.61553659e-02,4.39113900e-02,7.17036948e-02,5.36698010e-03,2.75546964e-03,1.34176388e-02,3.65323178e-03,1.05134444e-03,1.84575524e-02,7.79634388e-03,3.14921234e-03,8.63138810e-02,2.00516870e-03],[2.59744599e-02,1.14650966e-03,3.81580408e-04,5.11003425e-03,6.71650341e-04,1.00969505e-02,7.34422763e-04,2.01144870e-02,4.64319997e-03,4.63992881e-04,7.68486932e-02,3.51626775e-04,8.92823376e-03,4.07721102e-03,2.63475049e-02,1.68296248e-02,2.18672410e-01,2.34135558e-04,4.68556359e-02,4.34731226e-03,4.95799005e-01,2.62362696e-02,5.92098339e-04,4.54298686e-03],[5.43444902e-02,3.85460001e-03,1.28310523e-04,3.42131010e-04,2.55003804e-03,7.67760575e-02,1.76249305e-03,3.57429266e-01,2.29516421e-02,1.50239770e-03,9.14376527e-02,2.51506775e-04,8.69519636e-03,5.97129669e-03,2.21612630e-03,1.32660158e-02,7.44757336e-03,4.47088940e-04,9.08462517e-03,7.34946283e-04,6.66477159e-02,2.77752485e-02,6.21148502e-04,2.43762448e-01],[2.79144287e-01,7.47270370e-03,2.87666678e-01,2.20414758e-01,9.35743600e-02,6.64330611e-04,7.48921139e-03,9.52059403e-03,1.13979578e-02,3.89943039e-03,1.76754519e-02,6.14632445e-04,1.06477831e-02,1.39833912e-02,2.43927981e-03,2.05733860e-03,6.58773584e-03,6.80094119e-03,2.19317665e-03,1.72719127e-03,9.81951132e-03,8.14830186e-04,2.87963683e-03,5.14701533e-04],[4.72311629e-03,9.75260735e-02,3.54774692e-03,4.69116122e-03,2.14805342e-02,2.02830173e-02,2.13392571e-01,1.67965159e-01,5.37068173e-02,9.62795597e-03,4.28975970e-02,2.86245663e-02,2.42668428e-02,2.49059927e-02,7.70168528e-02,1.13182617e-02,7.36953504e-03,3.79361361e-02,7.61808231e-02,3.44199352e-02,6.89522643e-03,1.09352414e-02,5.14015974e-03,1.51486676e-02],[3.56739876e-03,1.64810084e-02,7.80417863e-03,3.30556254e-03,1.10729015e-03,1.78383291e-02,3.54794250e-03,2.22165952e-03,6.07590983e-03,1.62009744e-03,1.46913966e-02,5.80274343e-01,2.01594532e-02,4.06013392e-02,1.11170873e-01,2.88177822e-02,1.38677089e-02,4.99656517e-03,7.96697568e-03,8.29554200e-02,3.51749477e-03,2.34995913e-02,2.49714754e-03,1.41447480e-03],[4.27839579e-03,8.86270553e-02,1.84959196e-03,4.06610407e-03,7.81112118e-03,6.53478354e-02,1.53619200e-02,1.11455033e-02,4.66774032e-02,9.00656544e-03,2.17158999e-02,1.55334741e-01,2.17930432e-02,9.71777067e-02,1.16389960e-01,1.70494899e-01,1.34518929e-02,3.01370751e-02,7.97996391e-03,1.16146505e-02,7.57626723e-03,6.65063933e-02,8.57520476e-03,1.70807932e-02],[6.58174802e-04,1.23500172e-02,1.49263768e-04,4.02240362e-03,5.36250125e-04,5.92212193e-02,3.81240342e-03,1.02628265e-02,8.58247001e-03,2.22642217e-02,1.14922170e-02,9.57201328e-03,7.77601898e-02,1.39083108e-02,1.18074045e-01,1.89323314e-02,4.26913977e-01,2.59779976e-03,5.63665153e-03,1.38111100e-01,1.58033753e-03,3.17128263e-02,1.61180161e-02,5.73093863e-03],[7.18825636e-03,3.21167335e-02,4.04914515e-03,6.06475119e-03,5.57212457e-02,8.47206544e-03,1.31319212e-02,1.28426895e-01,5.95071018e-02,1.51102155e-01,1.00634033e-02,2.07743850e-02,1.75673306e-01,1.72919333e-01,2.33846400e-02,4.07011295e-03,3.41068022e-02,3.05017065e-02,2.59724888e-03,7.60000525e-03,6.78809267e-03,4.22211830e-03,2.71363277e-02,1.43821547e-02],[1.64391252e-03,1.00484993e-02,2.19655689e-03,2.13351380e-03,3.22087668e-04,1.68898329e-02,3.57621536e-03,1.37330976e-03,4.20220941e-03,1.02736056e-03,1.53439781e-02,2.02293806e-02,5.84582752e-03,3.90300225e-03,1.41182140e-01,2.80051343e-02,1.78318042e-02,1.96922128e-03,9.69793275e-03,6.96060181e-01,1.95222895e-03,1.33604920e-02,7.19472650e-04,4.85760742e-04],[9.65405535e-03,3.29491086e-02,6.54772297e-03,2.30680266e-03,6.76450692e-03,1.12925306e-01,5.95110934e-03,1.12411333e-02,6.29229695e-02,5.08766016e-03,7.73322582e-03,1.51184663e-01,1.45056257e-02,7.82834888e-02,1.37988389e-01,1.37723058e-01,1.03504267e-02,3.92355919e-02,4.27516596e-03,3.65728303e-03,1.25556374e-02,1.15660384e-01,1.40389949e-02,1.64577141e-02],[3.83761190e-02,3.68586695e-03,1.07333902e-03,1.20434789e-02,1.00729184e-03,1.27336998e-02,1.88539177e-03,4.38282155e-02,5.60065499e-03,1.80646416e-03,6.22060746e-02,3.03330948e-04,5.56435762e-03,3.69218714e-03,1.31198158e-02,1.29922424e-02,3.27862166e-02,2.55912193e-04,2.93461800e-01,1.11955460e-02,3.96275014e-01,3.34572047e-02,1.03213300e-03,1.16176754e-02],[3.34078982e-03,6.28096312e-02,1.46940164e-03,2.34168698e-03,1.23577733e-02,4.78785634e-02,8.57045408e-03,7.47908326e-03,4.97474596e-02,5.62233478e-03,1.26124769e-02,2.16587991e-01,1.98795293e-02,1.28707021e-01,1.01532951e-01,1.91556349e-01,8.29924084e-03,3.68827023e-02,2.38440675e-03,3.30269686e-03,7.26808468e-03,4.79670875e-02,8.10054690e-03,1.33018037e-02],[9.18043661e-04,3.00698839e-02,4.31341305e-03,1.46921324e-02,1.77434292e-02,5.95537154e-03,8.07873160e-02,4.90342686e-03,1.88376307e-02,8.12109653e-03,5.64094959e-03,8.28930363e-02,3.50453593e-02,1.05981663e-01,7.36039281e-02,9.54722986e-03,1.15408376e-02,4.10941809e-01,8.20438750e-03,8.51399358e-03,1.73258386e-03,1.66491996e-02,3.60956974e-02,7.26762693e-03],[3.22256307e-03,3.06595303e-02,1.14870328e-03,2.98809889e-03,1.66398510e-02,9.91241187e-02,3.30099650e-03,4.55998965e-02,6.59215376e-02,2.58375585e-01,8.53787176e-03,2.91840397e-02,7.27310553e-02,9.61181074e-02,3.14781331e-02,3.05065718e-02,4.85665910e-02,1.76508315e-02,1.48839736e-03,6.88043516e-03,4.45694290e-03,3.16835530e-02,5.79679832e-02,3.57685611e-02],[5.92093915e-03,6.95101917e-03,1.27137941e-03,3.10812495e-03,3.30329261e-04,1.28253981e-01,1.97673752e-03,1.15975756e-02,1.08727617e-02,1.16674288e-03,4.33802605e-02,4.92859259e-03,3.11586331e-03,2.66488385e-03,1.21330656e-01,1.23859756e-01,2.91178878e-02,5.92997472e-04,1.17845573e-01,4.23544273e-02,3.75221148e-02,2.46208578e-01,3.63528205e-04,5.52653559e-02],[9.02387220e-03,4.45474684e-02,1.34742865e-02,2.93018389e-02,4.37995279e-03,3.30342464e-02,1.11353770e-02,1.87999625e-02,2.53631324e-02,1.44580960e-01,3.90708633e-02,3.70082557e-02,1.82577774e-01,1.13176420e-01,3.53657752e-02,1.53947482e-02,1.30854443e-01,6.95111556e-03,2.03766651e-03,6.05848134e-02,1.54433371e-02,1.44588985e-02,1.06643019e-02,2.77050515e-03],[8.11556801e-02,1.16309188e-02,1.46261475e-03,1.06292255e-02,7.68748373e-02,2.61580292e-02,3.25942109e-03,9.91556495e-02,6.20787479e-02,5.86714875e-03,2.05197573e-01,1.12326664e-03,2.13951617e-02,2.33993027e-02,2.42321510e-02,4.60651331e-02,9.30750892e-02,2.80242949e-03,1.66924186e-02,6.37203129e-03,9.99583155e-02,4.73946594e-02,4.20989143e-03,2.98103429e-02],[2.52681263e-02,1.29481062e-01,6.34891773e-03,2.56456365e-03,3.04849356e-01,1.12840431e-02,1.19261518e-02,3.46237719e-02,1.76694155e-01,4.55323420e-03,1.86007060e-02,1.09687122e-02,5.70271630e-03,2.98284907e-02,5.11938445e-02,6.77255318e-02,2.24802596e-03,8.03138688e-02,2.06227135e-03,1.61297899e-03,6.01274893e-03,8.48407112e-03,1.81770325e-03,5.83488541e-03],[1.45583646e-03,2.34080236e-02,2.20399583e-03,7.88271800e-03,7.85065722e-03,5.42826438e-03,6.44405512e-03,6.34861737e-03,1.06443968e-02,6.74300045e-02,2.74100690e-03,1.34422883e-01,9.77159366e-02,2.34443843e-01,2.21050140e-02,6.51552249e-03,1.26907919e-02,4.53106947e-02,2.99055857e-04,2.96174036e-03,1.98276970e-03,6.71360409e-03,2.89873570e-01,3.12708481e-03],[4.53252811e-03,4.94381227e-02,1.76684698e-03,3.28511558e-03,2.77393889e-02,5.03909141e-02,2.85149715e-03,2.55544554e-03,1.31469071e-01,7.66826654e-03,8.40510335e-03,2.44884137e-02,4.34243586e-03,2.63085086e-02,1.28855675e-01,4.25171614e-01,5.22087002e-03,4.70283329e-02,1.87239540e-03,3.24025587e-03,3.29653174e-03,3.09050679e-02,3.24457558e-03,5.92306023e-03],[5.41618443e-04,1.87912286e-04,2.83434149e-03,9.48829353e-01,7.15385177e-05,2.06192402e-04,1.90848703e-04,7.03995975e-05,6.82758036e-05,1.18892443e-04,3.52908252e-03,9.67447268e-05,2.56648753e-04,1.35561670e-04,2.54118885e-03,8.97527265e-04,1.16919000e-02,5.49797369e-05,4.26007202e-03,2.06120145e-02,1.53884967e-03,1.07231166e-03,1.81350872e-04,1.24747112e-05],[3.13593484e-02,2.05375865e-04,6.06184499e-03,3.94597976e-03,8.20915593e-05,4.94006788e-03,1.08702741e-04,6.80845045e-03,2.01690989e-03,8.40331486e-04,3.11805471e-03,2.89414340e-04,1.49826426e-03,6.78283221e-04,4.21821093e-03,5.11847064e-03,2.62547135e-02,4.70591513e-05,1.49909884e-03,1.50231924e-03,8.90646160e-01,5.54353325e-03,2.66929972e-03,5.47949516e-04]],[[1.16198882e-03,4.02763858e-02,6.64035499e-04,5.05209416e-02,5.10606349e-01,1.10796456e-04,7.11985007e-02,2.21249252e-03,4.57387529e-02,1.49146229e-01,3.37537774e-03,6.94459246e-04,3.13775949e-02,1.78442709e-02,5.61912602e-04,1.94852124e-03,3.13834660e-03,6.41304627e-02,9.60689504e-05,1.27761363e-04,2.15465858e-04,7.79412949e-05,3.12946714e-03,1.64593174e-03],[7.23356068e-01,4.46583796e-03,9.37043577e-02,3.29109095e-02,7.99522211e-04,1.02613471e-03,2.62325234e-03,2.12054662e-02,1.42813819e-02,2.80019641e-02,8.40687286e-03,4.62317606e-03,1.49894226e-02,2.65355948e-02,1.61202654e-04,4.18531796e-04,1.26696343e-03,1.75107183e-04,8.52539597e-05,3.20781954e-03,1.73503498e-03,7.23830424e-04,1.36476327e-02,1.64866506e-03],[5.07699745e-03,4.09931294e-04,1.04750070e-04,3.84924281e-03,2.49628560e-04,4.93025000e-04,1.07038755e-03,1.43283363e-02,4.43033315e-03,1.19758690e-04,2.94995159e-01,1.73502431e-05,8.80309287e-03,1.70534046e-03,1.13686000e-03,3.32213659e-03,4.28152978e-01,2.35461102e-05,1.71861406e-02,7.40025775e-04,2.05937371e-01,6.80279126e-03,7.63902353e-05,9.68458946e-04],[1.25889322e-02,1.20302360e-03,1.08377753e-05,5.72497629e-05,2.77261744e-04,1.17392680e-02,1.61721720e-03,4.60221946e-01,2.41068564e-02,2.46416341e-04,7.19546005e-02,1.29998125e-05,4.44107503e-03,2.07658554e-03,2.82034525e-05,2.99329474e-03,2.34084902e-03,2.38361827e-05,8.08645331e-04,3.31790252e-05,7.57955294e-03,5.17688692e-03,3.46352972e-05,3.90426636e-01],[4.33306932e-01,5.99223073e-04,4.12810117e-01,1.21950693e-01,6.04777923e-03,1.28113124e-05,8.08588695e-03,4.45607671e-04,1.88317511e-03,1.00969242e-04,6.24072319e-03,3.04495206e-05,3.97861423e-03,1.66894961e-03,3.14738209e-05,1.34259288e-04,8.75720987e-04,3.16708145e-04,1.24302125e-04,5.17198678e-05,1.04693451e-03,1.00208046e-04,1.06400563e-04,5.02671610e-05],[2.91191833e-03,5.25753573e-02,1.03049132e-03,1.32730347e-03,2.06543785e-02,2.48988252e-02,1.20860308e-01,3.09386194e-01,1.81688204e-01,6.51946897e-03,3.05930395e-02,1.37311416e-02,1.70254111e-02,1.12267593e-02,5.40548787e-02,6.55508367e-03,5.29962312e-03,1.77974980e-02,8.91730338e-02,6.48814300e-03,5.97828114e-03,5.36755519e-03,1.22941553e-03,1.36275981e-02],[3.14941775e-04,3.56584066e-03,4.18916810e-03,4.43210360e-04,1.74212895e-04,1.15576722e-02,7.95912230e-04,1.10571928e-04,2.31934246e-03,3.08224699e-04,1.01129375e-02,8.78731966e-01,5.55580435e-03,2.22583823e-02,1.07719693e-02,1.38535481e-02,3.14055919e-03,1.20104512e-03,1.41315663e-03,1.69236101e-02,6.65862462e-04,1.07410457e-02,4.95651795e-04,3.55436961e-04],[1.07318326e-03,5.24029881e-02,6.91750727e-04,1.09406817e-03,1.07511994e-03,1.10532172e-01,7.46611832e-03,1.56355347e-03,6.90457821e-02,3.02045769e-03,1.56808533e-02,1.95695743e-01,1.30687263e-02,1.33762360e-01,3.61075215e-02,2.31805623e-01,6.99053705e-03,1.00293281e-02,1.41928892e-03,1.58353220e-03,2.61619175e-03,8.71865973e-02,2.17013014e-03,1.39183281e-02],[1.25556675e-04,1.64971612e-02,1.75473961e-05,8.64869275e-04,1.22811776e-04,7.98691437e-02,2.94199842e-03,6.66983658e-03,3.41520756e-02,2.11876519e-02,1.37916738e-02,3.74183850e-03,1.00637019e-01,2.00801957e-02,5.43135405e-02,3.46687362e-02,5.12464762e-01,6.37474994e-04,1.08633202e-03,4.68126573e-02,2.35912768e-04,2.77504437e-02,1.33494148e-02,7.98134785e-03],[1.60477236e-02,2.50049010e-02,3.06479982e-03,6.56849938e-03,4.07652147e-02,2.34686583e-03,2.06984356e-02,1.18172221e-01,1.77953646e-01,1.30404428e-01,1.82410423e-02,7.51322974e-03,2.00322419e-01,1.65262654e-01,2.88224337e-03,1.10038032e-03,2.78201904e-02,8.38288292e-03,1.19507709e-03,3.97960935e-03,2.33314489e-03,9.10519506e-04,8.94569233e-03,1.00841550e-02],[4.52225911e-04,5.09865256e-03,2.96378788e-03,3.01690714e-04,3.69010995e-05,2.63906978e-02,1.66348042e-03,2.53479375e-04,3.74150695e-03,2.28553996e-04,3.31070721e-02,1.17785204e-02,4.58772155e-03,1.94487139e-03,4.15754430e-02,2.53500175e-02,5.91657963e-03,4.36662260e-04,2.09452212e-03,8.24302435e-01,3.23978486e-04,7.31580378e-03,9.70474866e-05,3.83975021e-05],[3.36119044e-03,2.48565115e-02,9.26442211e-04,5.66587551e-04,1.78405677e-03,2.20345333e-01,3.49871838e-03,2.73758685e-03,9.17309076e-02,2.66266940e-03,7.98286404e-03,3.75907794e-02,6.86708186e-03,9.45661366e-02,1.04023162e-02,3.01945388e-01,6.46818941e-03,9.00866929e-03,5.30501944e-04,4.19808348e-04,6.15590345e-03,1.43427148e-01,5.61342528e-03,1.65518131e-02],[2.49246731e-02,1.76632602e-03,3.22650594e-04,5.51268831e-03,3.01222171e-04,3.46472114e-03,1.79896737e-03,7.21375421e-02,1.02418289e-02,8.07718432e-04,1.40521407e-01,4.90278253e-05,3.81196383e-03,2.46161851e-03,1.30564370e-03,6.80008577e-03,3.64878401e-02,3.83568877e-05,1.29046902e-01,1.14334049e-03,4.60679710e-01,7.40854964e-02,2.57695705e-04,2.20325086e-02],[3.34092323e-03,4.74026538e-02,1.17686426e-03,1.32117036e-03,6.52293861e-03,4.65165190e-02,9.75126866e-03,4.78480663e-03,1.14811748e-01,2.62081041e-03,1.94736607e-02,1.63709030e-01,1.62104759e-02,2.00910419e-01,2.42087394e-02,2.45969802e-01,6.84754970e-03,2.22338866e-02,8.23171809e-04,5.25012554e-04,6.49226131e-03,3.38562019e-02,1.75918476e-03,1.87309179e-02],[5.50951401e-04,8.50319304e-03,2.97367293e-03,7.11999321e-03,4.21203813e-03,4.08725860e-03,1.03940561e-01,8.10542377e-04,3.62209827e-02,2.40507000e-03,3.99492588e-03,1.09731928e-01,3.78328077e-02,1.20319493e-01,3.68607938e-02,8.90340097e-03,1.05612325e-02,4.53492403e-01,3.19921062e-03,9.03024047e-04,9.13157244e-04,1.90653130e-02,1.54722687e-02,7.92579073e-03],[2.41289521e-03,1.93363689e-02,5.49921606e-05,3.88877292e-04,2.48878752e-03,8.64500701e-02,6.66120730e-04,2.38410458e-02,1.43808842e-01,2.84423023e-01,4.33980813e-03,6.57944055e-03,6.52113706e-02,2.21674442e-01,7.14099267e-04,1.76886786e-02,1.96589977e-02,1.44784700e-03,2.54727402e-05,7.27653212e-04,4.89696569e-04,4.81131533e-03,3.21413167e-02,6.06188923e-02],[1.33330270e-03,2.04043090e-03,3.12452001e-04,8.89766321e-04,7.99454283e-05,5.24059124e-02,2.17710709e-04,6.74937246e-03,1.46736279e-02,4.08238353e-04,3.80084254e-02,7.39994051e-04,8.02399998e-04,1.24555780e-03,9.43535194e-02,3.19080174e-01,1.66683812e-02,1.87918529e-04,5.23606502e-02,8.65899492e-03,1.18139433e-02,3.48087937e-01,5.13808045e-05,2.88299937e-02],[7.25058792e-03,3.08497753e-02,5.38907666e-03,1.09636597e-02,9.03276668e-04,2.22525466e-02,7.97763653e-03,2.72375345e-03,6.98115379e-02,2.14217171e-01,5.30506261e-02,1.49649307e-02,2.57221073e-01,2.38668516e-01,9.30319366e-04,3.46218748e-03,3.38735804e-02,7.20395357e-04,1.48172534e-04,6.07814826e-03,2.88435980e-03,5.32903755e-03,5.12288185e-03,5.20674745e-03],[1.88047662e-02,1.64751224e-02,6.07483205e-04,9.90496948e-03,7.62037560e-02,7.44426716e-03,3.40823410e-03,4.25296053e-02,7.24382997e-02,8.20222590e-03,3.14458013e-01,3.86235741e-04,2.63716262e-02,2.51159072e-02,3.39913578e-03,5.66008389e-02,2.35265777e-01,9.15125012e-04,4.26197518e-03,2.41006748e-03,3.69689539e-02,2.29579173e-02,2.22297036e-03,1.26466760e-02],[2.00811401e-02,1.48420304e-01,2.83596013e-03,1.17262278e-03,2.22997472e-01,1.37714446e-02,9.42476001e-03,1.31330993e-02,3.42639118e-01,2.81427824e-03,1.17099006e-02,2.93455669e-03,4.95570712e-03,2.08143834e-02,1.00741209e-02,9.60139781e-02,1.72056933e-03,5.95297068e-02,4.99450136e-04,2.19262962e-04,3.80704482e-03,5.28346701e-03,3.41333391e-04,4.80631320e-03],[4.74819011e-04,1.73536874e-02,3.14160337e-04,2.49298592e-03,1.81293814e-03,2.05674488e-03,6.44842722e-03,4.28873609e-04,1.54865962e-02,2.84019522e-02,3.34518938e-03,2.21766546e-01,1.04405403e-01,3.89439404e-01,1.77055458e-03,4.56413906e-03,5.70894824e-03,1.03210900e-02,2.35058160e-05,1.80437739e-04,5.50318742e-04,3.06999660e-03,1.74767524e-01,4.81576147e-03],[1.99511694e-03,3.62332650e-02,1.38931337e-03,1.10587012e-03,5.09103807e-03,8.35244879e-02,1.53348199e-03,3.20160470e-04,1.17309950e-01,2.87874462e-03,5.28824329e-03,2.26638261e-02,3.82875837e-03,4.11401503e-02,2.99951471e-02,5.77955782e-01,3.65890958e-03,2.75283828e-02,3.72622308e-04,7.54800276e-04,9.50222195e-04,3.08431238e-02,1.26688252e-03,2.37174681e-03],[4.89888771e-05,5.59870277e-05,7.89148547e-03,9.79511857e-01,1.15232824e-05,8.08132609e-06,3.87073669e-04,5.56718578e-06,2.08356851e-05,6.92895410e-05,5.65673364e-03,2.72065463e-05,9.07939684e-05,2.08143538e-05,6.70644658e-05,7.71660780e-05,7.36506132e-04,8.83833491e-06,1.93926785e-03,2.84157810e-03,2.30075864e-04,2.76183477e-04,1.53611327e-05,1.71539375e-06],[2.64301803e-02,2.29062760e-04,3.85787594e-03,3.34309600e-03,1.81299256e-05,1.94659736e-03,2.15314940e-04,7.41211418e-03,6.42902078e-03,2.52288155e-04,4.09199223e-02,9.45772917e-05,1.30318035e-03,5.55740204e-04,3.38054699e-04,7.68623548e-03,2.37849727e-02,5.59694763e-06,3.64061591e-04,1.46304280e-03,8.48297417e-01,2.28875559e-02,1.55347609e-03,6.12389122e-04]],[[2.73788488e-03,5.24781868e-02,3.00207548e-03,1.79658502e-01,1.99985728e-01,1.90873485e-04,3.37464772e-02,8.21514335e-03,2.56920122e-02,1.73686281e-01,6.84333779e-03,1.43637857e-03,1.32187128e-01,4.50980254e-02,2.51843594e-03,9.12081276e-04,4.13248176e-03,1.13796733e-01,1.13810727e-03,5.88519615e-04,1.07397733e-03,4.56522161e-04,9.16044414e-03,1.26463652e-03],[2.58210599e-01,2.38810293e-03,2.41513148e-01,3.57672542e-01,9.82708763e-04,3.02617234e-04,2.85524130e-03,1.35057699e-02,1.16145443e-02,1.63848400e-02,6.50100317e-03,3.40301567e-03,1.69213284e-02,3.99193168e-02,2.39952205e-04,1.87206737e-04,1.81023811e-03,9.06067726e-04,2.91331555e-04,4.33253730e-03,2.29009846e-03,2.60081812e-04,1.72679033e-02,2.39862289e-04],[2.61157914e-03,2.55597406e-04,3.70413727e-05,1.63384643e-03,1.20177363e-04,4.35297785e-04,1.69657476e-04,6.79422216e-03,2.66454322e-03,9.50804533e-05,1.03228480e-01,9.60325087e-06,3.41843511e-03,7.73918058e-04,1.86420826e-03,8.14153999e-03,3.84597838e-01,3.28081660e-05,1.06583126e-02,4.40568838e-04,4.59440708e-01,1.19055798e-02,1.84495002e-04,4.86358535e-04],[1.32739162e-02,8.24958924e-03,4.71349849e-05,4.65454155e-04,3.78390797e-03,1.09408572e-02,2.48684594e-03,3.67943943e-01,2.76131518e-02,1.92510011e-03,2.10373849e-01,3.95866955e-05,2.75229570e-02,7.17563601e-03,7.72191444e-04,9.74624790e-03,9.42378491e-03,1.41686760e-04,4.20765346e-03,1.90125444e-04,1.38709813e-01,3.20771076e-02,9.79011995e-04,1.21910505e-01],[1.55030042e-01,1.11641316e-03,3.31703305e-01,4.47616726e-01,9.66550410e-03,4.25600556e-05,3.33233899e-03,1.25578395e-03,3.47112375e-03,3.74967203e-04,1.84688400e-02,3.91840840e-05,5.59483282e-03,2.55126297e-03,2.42789247e-04,5.95400459e-04,1.04402322e-02,2.73092627e-03,1.09728926e-03,4.31422464e-04,3.46143311e-03,4.66374127e-04,2.41341797e-04,2.99521053e-05],[3.71631724e-03,9.22411308e-02,1.06527295e-03,2.85766670e-03,1.04168586e-01,2.00200137e-02,4.57265563e-02,2.26125166e-01,7.45199695e-02,6.60471022e-02,2.84451488e-02,8.11575912e-03,2.88572721e-02,3.28975655e-02,1.16238639e-01,4.00669500e-03,3.83546879e-03,1.42821018e-02,6.18869737e-02,7.14911520e-03,1.01658190e-02,6.02718676e-03,2.68841404e-02,1.47203794e-02],[2.73902598e-03,4.92881238e-03,9.79699846e-03,2.44695460e-03,8.42825684e-04,7.74764735e-03,1.28897198e-03,8.84976937e-04,3.05507123e-03,7.75187742e-04,2.21391004e-02,7.38660932e-01,7.24701071e-03,4.89017665e-02,3.83439958e-02,1.16605442e-02,6.39374228e-03,5.64128114e-03,6.85617141e-03,5.29740080e-02,2.90540513e-03,1.73412655e-02,6.06947951e-03,3.58814636e-04],[2.65938416e-03,7.00922087e-02,8.03548261e-04,3.35796177e-03,5.76715125e-03,7.45606273e-02,1.04252491e-02,3.91339650e-03,4.45685498e-02,1.95474066e-02,1.65140089e-02,1.00561321e-01,2.06383709e-02,1.72405198e-01,6.20088428e-02,1.71857655e-01,8.77679698e-03,3.39937843e-02,3.15748248e-03,2.74317153e-03,7.39175826e-03,1.18965752e-01,3.26981954e-02,1.25922523e-02],[1.73336448e-04,2.13685934e-03,2.59943299e-05,1.96120865e-03,1.14214505e-04,1.75261274e-02,1.03189377e-03,4.31896513e-03,6.23436086e-03,2.82519218e-03,7.31600914e-03,2.00740318e-03,1.65930167e-02,8.90126452e-03,7.64941499e-02,8.61052796e-02,6.52656674e-01,1.44177431e-03,1.17799104e-03,5.21802641e-02,4.70957631e-04,3.45414914e-02,2.27912106e-02,9.74372902e-04],[5.35270665e-03,2.84525771e-02,5.09897666e-03,2.33953539e-02,3.74943353e-02,2.85368413e-03,3.98410950e-03,1.06225692e-01,9.82729867e-02,3.06794524e-01,1.87156703e-02,5.55135682e-03,1.23335727e-01,1.17202736e-01,7.10363872e-03,1.91861170e-03,4.06879447e-02,1.55701367e-02,1.87061459e-03,6.81651337e-03,6.04684604e-03,1.93612662e-03,3.23345475e-02,2.98462342e-03],[2.78474251e-03,3.49955144e-03,5.03111864e-03,1.70198421e-03,3.59691738e-04,2.08097547e-02,3.19659384e-03,1.70796877e-03,4.32934286e-03,5.54055965e-04,4.17965017e-02,2.19823364e-02,3.00707296e-03,6.84907800e-03,1.48550138e-01,3.39273587e-02,1.56410784e-02,4.33415966e-03,7.08735222e-03,6.57931745e-01,1.65251142e-03,1.06312260e-02,2.43366393e-03,2.00909140e-04],[5.88473259e-03,2.05694754e-02,1.09711010e-03,9.47980792e-04,5.81031339e-03,1.01581834e-01,3.67299630e-03,8.72514211e-03,7.85892084e-02,8.40275083e-03,6.81285001e-03,3.56570370e-02,1.32151535e-02,1.16496973e-01,4.38412465e-02,2.41606623e-01,7.48141622e-03,3.68547067e-02,1.26308564e-03,5.89530391e-04,1.77422222e-02,1.97865263e-01,3.19668017e-02,1.33255469e-02],[4.56753606e-03,4.06320626e-03,1.98608133e-04,8.45032558e-03,1.53093494e-03,1.03968079e-03,1.16962264e-03,2.06895247e-02,4.95174527e-03,3.46355280e-03,6.98362365e-02,2.85160258e-05,6.25592889e-03,2.50347308e-03,4.55006119e-03,5.91902202e-03,1.62521936e-02,1.05793595e-04,4.45088178e-01,4.42496501e-03,3.74001205e-01,1.54828839e-02,2.08311505e-03,3.34365712e-03],[6.08563051e-03,8.28282535e-02,1.38211029e-03,2.86427420e-03,1.35573531e-02,3.12522762e-02,9.20266472e-03,1.15448525e-02,8.83562043e-02,1.72703881e-02,1.89497676e-02,8.88175219e-02,4.32661250e-02,2.68001914e-01,4.49870415e-02,1.01164065e-01,7.71839172e-03,5.48795387e-02,2.10961001e-03,1.01343682e-03,1.54100824e-02,5.88342212e-02,1.48529820e-02,1.56512149e-02],[1.36466988e-03,1.71446279e-02,3.08887684e-03,2.20954735e-02,1.92836132e-02,3.56915686e-03,4.18216996e-02,2.51770904e-03,2.70841829e-02,2.49440372e-02,7.60226510e-03,4.61335443e-02,5.66848144e-02,1.13154463e-01,7.45343864e-02,7.28732441e-03,9.77565721e-03,4.42211092e-01,9.52395424e-03,3.32466629e-03,2.26483960e-03,1.79899391e-02,4.04791608e-02,6.11987105e-03],[8.31546960e-04,1.86532438e-02,1.05394080e-04,1.18332368e-03,3.17618996e-03,3.23281102e-02,5.94779325e-04,2.03792322e-02,6.58244267e-02,4.37111199e-01,3.82823264e-03,4.05449746e-03,9.68039781e-02,1.69746876e-01,2.66280328e-03,2.04083771e-02,2.19387356e-02,4.19508899e-03,8.74675607e-05,1.09990546e-03,8.85470072e-04,7.59620918e-03,6.96668327e-02,1.68378893e-02],[3.42373014e-03,2.53515947e-03,2.74333928e-04,1.70773244e-03,8.30815174e-04,4.68552560e-02,3.64554493e-04,1.14766676e-02,1.00538768e-02,2.16647703e-03,3.65095846e-02,8.08613608e-04,1.68554601e-03,2.82658497e-03,1.38823733e-01,2.30040401e-01,2.92479545e-02,2.82326335e-04,5.03519997e-02,1.30389072e-02,5.55757470e-02,3.23767036e-01,2.02152738e-03,3.53314243e-02],[5.56648150e-03,2.36693174e-02,1.28430817e-02,5.68852834e-02,2.00279849e-03,6.19291002e-03,7.04912329e-03,1.25383083e-02,4.33222540e-02,3.22321713e-01,5.01850098e-02,9.43256915e-03,1.64702162e-01,1.78657308e-01,2.30761012e-03,3.22764018e-03,5.03264517e-02,4.51864116e-03,7.64684286e-04,1.32810641e-02,8.89844913e-03,3.18375346e-03,1.69261824e-02,1.19717175e-03],[8.09432659e-03,9.39804222e-03,2.12304541e-04,1.42736891e-02,2.90612727e-02,4.75492515e-03,1.03279168e-03,2.21737921e-02,4.31239866e-02,6.57437462e-03,1.84478387e-01,7.72515486e-05,3.10733002e-02,1.26386806e-02,4.06052778e-03,1.65566429e-01,3.88753235e-01,5.95719321e-04,1.57237891e-03,1.56676583e-03,3.68540920e-02,2.47042198e-02,4.55459859e-03,4.80491994e-03],[2.42941249e-02,1.76949769e-01,4.19307686e-03,2.51833536e-03,2.31245592e-01,6.34362642e-03,7.03692762e-03,2.60256082e-02,1.49061099e-01,1.85746197e-02,1.09759895e-02,4.90061427e-03,1.97110772e-02,6.73614144e-02,2.94037294e-02,3.31496112e-02,1.84020412e-03,1.57510474e-01,1.80935732e-03,3.97057272e-04,8.92723352e-03,9.60806292e-03,3.46744037e-03,4.69486834e-03],[3.18518840e-04,7.47617939e-03,6.62455044e-04,1.19225122e-02,8.25650350e-04,2.30334350e-03,2.95051979e-03,4.74912173e-04,1.00880060e-02,7.85763413e-02,1.09716214e-03,8.34160671e-02,7.63686821e-02,2.86890417e-01,3.03337607e-03,3.89600149e-03,8.39024037e-03,2.34759655e-02,7.17397998e-05,4.69212333e-04,6.23546366e-04,9.64309182e-03,3.85115176e-01,1.91094237e-03],[1.07167056e-02,5.54740466e-02,3.98199772e-03,3.66386212e-03,2.54046917e-02,5.60458526e-02,2.19094800e-03,5.05993748e-03,1.31957859e-01,2.61293184e-02,1.23847649e-02,1.79944150e-02,9.46589652e-03,7.03440905e-02,1.01239800e-01,2.40805671e-01,5.26929088e-03,1.52034640e-01,2.66725826e-03,1.85144402e-03,6.21510157e-03,4.87741120e-02,6.29770150e-03,4.03068028e-03],[6.54384567e-05,2.43395389e-05,1.85144972e-03,9.77065980e-01,1.83558986e-05,1.33241901e-05,9.14137418e-05,1.11502250e-05,1.73404387e-05,6.43730164e-05,3.50024900e-03,9.69637495e-06,4.13962007e-05,3.52181414e-05,1.52857974e-04,2.90754804e-04,3.61578213e-03,1.40371203e-05,2.74391682e-03,9.34312120e-03,5.79278101e-04,2.86989729e-04,1.62265540e-04,1.07056906e-06],[5.18478313e-03,3.84015693e-05,8.56964034e-04,1.15470623e-03,1.66021400e-05,2.99927313e-04,5.27660159e-05,4.03568428e-03,2.29852414e-03,7.13902409e-05,5.98689914e-03,1.53293022e-05,1.98110210e-04,1.87304206e-04,3.45438020e-04,7.07423640e-03,2.86888275e-02,6.28179259e-06,3.47416964e-04,5.23505732e-04,9.38557446e-01,3.39080184e-03,6.39016449e-04,2.97318511e-05]]
    ])

    plot_confusion_matrices(predictionsV12)
