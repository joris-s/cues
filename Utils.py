import random
import cv2
import numpy as np
import tensorflow as tf
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
if os.name != 'nt':
    matplotlib.use('tkagg')

# Set font parameters for Matplotlib
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

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
TRAIN_FOLDER = 'data/self/joris'
VAL_FOLDER = 'data/self/ercan'
TEST_FOLDER = 'data/self/roos'
META_TRAIN_FOLDER = 'data/UCF-101/train'
META_VAL_FOLDER = 'data/UCF-101/val'
AL_FOLDER = 'data/slapi/active-learning'
if (os.name == 'nt') == False:
    LABELED_FOLDER = 'data/slapi/labeled'
    UNLABELED_FOLDER = 'data/slapi/unlabeled'
    TRAIN_FOLDER = 'data/slapi/train'
    VAL_FOLDER = 'data/slapi/val'
    TEST_FOLDER = 'data/slapi/test'
    META_TRAIN_FOLDER = 'data/UCF-101/train'
    META_VAL_FOLDER = 'data/UCF-101/val'
    AL_FOLDER = 'data/slapi/active-learning'

LABEL_NAMES = sorted(os.listdir(TRAIN_FOLDER))

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

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

def filter_examples_per_class(examples_list, max_examples_per_class):
    num_examples_per_class = {}
    filtered_list = []
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
        self.old_pairs = -1

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*' + self.extension))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))
        if self.old_pairs == -1:
            self.old_pairs = pairs

        assert pairs == self.old_pairs
        self.old_pairs = pairs.copy()

        if self.shots != -1:
            pairs = filter_examples_per_class(pairs, self.shots)

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames, output_size=(self.resolution, self.resolution), frame_step=self.frame_step)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, tf.cast(label, tf.int32), path.__str__()

class ProposalGenerator:
    def __init__(self, model, path, n_frames, resolution, frame_step, extension='.mp4'):
        self.model = model
        self.path = path
        self.n_frames = n_frames
        self.output_size = (resolution, resolution)
        self.extension = extension
        self.frame_step = frame_step

    def process_frames(self, src):
        frames = []
        try:
            for _ in range(self.n_frames - 1):
                for _ in range(self.frame_step):
                    ret, frame = src.read()
                if ret:
                    frame = format_frames(frame, self.output_size)
                    frames.append(frame)
                else:
                    frames.append(np.zeros_like(frames[0]))
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
        starting_frame = 0

        while starting_frame < (total_frames - self.n_frames):
            processed_frames, start_index, stop_index = self.sliding_frames_from_video(starting_frame, cap)
            if isinstance(processed_frames, np.ndarray) == False:
                break
            starting_frame = stop_index + 1
            yield processed_frames, tf.cast(start_index, tf.int32), tf.cast(stop_index, tf.int32)


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

def cm_heatmap(actual, predicted, labels, savefigs=False, name='heatmap'):
    plt.clf()
    cm_num = confusion_matrix(actual, predicted)
    cm = []
    for i in range(len(cm_num)):
        row = cm_num[i]
        cm.append([(round(x/sum(row), 2)) for x in row])
    
    ax = sns.heatmap(cm, annot=True, cmap='BuPu', vmin=0.00, vmax=1.00)
    sns.set(rc={'figure.figsize':(15, 15)})
    sns.set(font_scale=1.9)
    ax.xaxis.tick_bottom()
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    
    if savefigs:
        plt.savefig('figs/cm/'+name+'.png', bbox_inches='tight', dpi=600)
    plt.close()    
    plt.clf()
    
# Modified plot_train_val function
def plot_train_val(history, title, savefigs=False):
    # Create figure and axis objects
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot train and validation loss
    ax[0].plot(history['loss'], label='Train', marker='o')
    ax[0].plot(history['val_loss'], label='Validation', marker='^')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim([0, 3])  # Adjust the y-axis limits
    ax[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    # Plot train and validation accuracy
    ax[1].plot(history['accuracy'], label='Train', marker='o')
    ax[1].plot(history['val_accuracy'], label='Validation', marker='^')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim([0, 1])  # Adjust the y-axis limits
    ax[1].set_xticks(np.arange(len(history['accuracy'])))
    ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    # Add legend and title
    ax[0].legend(loc='best', fontsize='large')
    ax[1].legend(loc='best', fontsize='large')
    fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    if savefigs:
        plt.savefig('figs/loss/'+title+'.png', dpi=600)
    plt.close() 
    plt.clf()
    
def get_class_weights(ds):
    labels = [int(label) for _, label in ds.unbatch()]
    total_labels = len(labels)
    class_weights = {i: 1/(labels.count(i)/total_labels) for i in range(len(list(set(labels))))}
    class_weight_sum = sum(class_weights.values())
    class_weight = {k: v/class_weight_sum for k, v in class_weights.items()}
    return class_weight
    
"""*****************************************
*             MoViNet Helpers              *
*****************************************"""
    
def AIPCreateBackboneAndClassifierModel(model_id, num_classes, frames_number, batch_size, resolution, 
                                        train_whole_model, dropout,
                                        checkpoint_dir,
                                        conv_type: str = '3d', se_type: str = '3d', activation: str = 'swish',
                                        gating_activation: str = 'sigmoid', stream_mode=False, load_pretrained_weights=True, training=True):
  '''
  Create video analysis model
  Return: movinet model
  //Andrzej Mąka, Aiseclab Sp. z o.o /POLAND //
  '''
  tf.keras.backend.clear_session()
  backbone = movinet.Movinet(model_id=model_id,
                            causal=stream_mode,
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
        dropout_rate = dropout
        )
  
  model.build([batch_size, frames_number, resolution, resolution, 3])
  return model
