import tqdm
import random
import cv2
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
params = {"font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from sklearn.metrics import confusion_matrix

OUTPUT_SIGNATURE = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16),
                    tf.TensorSpec(shape = (), dtype = tf.string))

MOVINET_PARAMS = {
    'a0': (172, 5),
    'a1': (172, 5),
    'a2': (224, 5),
    'a3': (256, 12),
    'a4': (290, 8),
    'a5': (320, 12)
    }

LABEL_NAMES = ['blinking', 'frown', 'fussy', 'grimace', 'hand', 'looking', 'mouth open', 'nothing', 'rolling', 'smile', 'staring', 'tongue out']

META_CLASSES = 20

def format_frames(frame, output_size):
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size, frame_step = 15):
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

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
  def __init__(self, path, n_frames, resolution, training = False, extension='.mp4', shots=-1, frame_step=15):

    self.path = path
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
    video_paths = list(self.path.glob('*/*'+self.extension))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()
    
    pairs = list(zip(video_paths, classes))
    if self.old_pairs == -1:
        self.old_pairs = pairs
        
    assert pairs == (self.old_pairs)
    self.old_pairs = pairs.copy()
    
    if self.shots != -1:
        pairs = filter_examples_per_class(pairs, self.shots)

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames, output_size=(self.resolution, self.resolution), frame_step=self.frame_step) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label, path.__str__()
      

def sliding_frames_from_video(path, n_frames, start_frame, step, output_size = (224,224)):
    
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(n_frames):
        for _ in range(step):
            ret, frame = cap.read()
        if ret:
          frame = format_frames(frame, output_size)
          frames.append(frame)
        else:
          frames.append(np.zeros_like(frames[0]))
    cap.release()
    frames = np.array(frames)[..., [2, 1, 0]]
    
    return frames

class SlidingFrameGenerator:
    
    def __init__(self, path, start_frame, n_frames, step = 5):
        self.path = path
        self.start_frame = start_frame
        self.n_frames = n_frames
        self.step = step
        
    def __call__(self):
        
        frames = sliding_frames_from_video(self.path, self.n_frames, self.start_frame, self.step)
        label = 0
        yield frames, label

def get_actual_predicted_labels(dataset, model):
    """
      Create a list of actual ground truth values and the predictions from the model.
    
      Args:
        dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.
    
      Return:
        Ground truth and predicted values for a particular dataset.
    """
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)
    
    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)
    
    return actual, predicted

def plot_confusion_matrix(actual, predicted, labels, ds_type, acc, model_id, model_type):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title(f'Confusion matrix of action recognition for {model_id}_{model_type} - %.2f Acc' % acc)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    plt.clf()
    
def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

def remove_paths(dataset1, dataset2):
    # Define a function to select only the frames and labels from each element of the dataset
    def select_frames_and_labels(frames, labels, paths):
        return frames, labels

    # Apply the function to each element of the dataset
    dataset1 = dataset1.map(select_frames_and_labels)
    dataset2 = dataset2.map(select_frames_and_labels)

    return dataset1, dataset2

def cm_heatmap(actual, predicted, labels, savefigs=False, name='heatmap'):
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
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    
    if savefigs:
        plt.savefig('figs/'+name+'.png', bbox_inches='tight', dpi=600)
        
    #plt.show()
    
# Read and process a video
def load_gif(file_path, image_size=(224, 224)):
  """Loads a gif file into a TF tensor.

  Use images resized to match what's expected by your model.
  The model pages say the "A2" models expect 224 x 224 images at 5 fps

  Args:
    file_path: path to the location of a gif file.
    image_size: a tuple of target size.

  Returns:
    a video of the gif file
  """
  # Load a gif file, convert it to a TF tensor
  raw = tf.io.read_file(file_path)
  video = tf.io.decode_gif(raw)
  # Resize the video
  video = tf.image.resize(video, image_size)
  # change dtype to a float32
  # Hub models always want images normalized to [0,1]
  # ref: https://www.tensorflow.org/hub/common_signatures/images#input
  video = tf.cast(video, tf.float32) / 255.
  return video


def preprocess_frame(frame, target_size=(224, 224)):
    # Resize the frame
    frame = cv2.resize(frame, target_size)
    # Normalize pixel values
    frame = frame / 255.
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

def load_video(file_path, target_size=(224, 224)):
    # Open video file
    cap = cv2.VideoCapture(file_path)
    # Read video frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess frame
        frame = preprocess_frame(frame, target_size)
        # Add to frames list
        frames.append(frame)
    # Close video file
    cap.release()
    # Stack frames into tensor
    video = np.concatenate(frames, axis=0)
    #video = tf.constant(frames, dtype=tf.float32, shape=[None, None, None, None, 3])
    #video.set_shape([-1, -1, -1, -1, 3])
    return video

# Get top_k labels and probabilities
def get_top_k(probs, label_map, k=12):
  """Outputs the top k model labels and probabilities on the given video.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k labels and probabilities.
  """
  # Sort predictions to find top_k
  # top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
  # collect the labels of top_k predictions
  # top_labels = tf.gather(label_map, top_predictions, axis=-1)
  # decode lablels
  # top_labels = [label.decode('utf8') for label in top_labels.numpy()]
  # top_k probabilities of the predictions
  # top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
  
  combined = list(zip(probs, label_map))
  sorted_tuples = sorted(combined, key=lambda x: x[0], reverse=True)
  
  return sorted_tuples

# Get top_k labels and probabilities predicted using MoViNets streaming model
def get_top_k_streaming_labels(probs, label_map, k=5):
  """Returns the top-k labels over an entire video sequence.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k probabilities, labels, and logit indices
  """
  top_categories_last = tf.argsort(probs, -1, 'DESCENDING')[-1, :1]
  # Sort predictions to find top_k
  categories = tf.argsort(probs, -1, 'DESCENDING')[:, :k]
  categories = tf.reshape(categories, [-1])

  counts = sorted([
      (i.numpy(), tf.reduce_sum(tf.cast(categories == i, tf.int32)).numpy())
      for i in tf.unique(categories)[0]
  ], key=lambda x: x[1], reverse=True)

  top_probs_idx = tf.constant([i for i, _ in counts[:k]])
  top_probs_idx = tf.concat([top_categories_last, top_probs_idx], 0)
  # find unique indices of categories
  top_probs_idx = tf.unique(top_probs_idx)[0][:k+1]
  # top_k probabilities of the predictions
  top_probs = tf.gather(probs, top_probs_idx, axis=-1)
  top_probs = tf.transpose(top_probs, perm=(1, 0))
  # collect the labels of top_k predictions
  top_labels = tf.gather(label_map, top_probs_idx, axis=0)
  # decode the top_k labels
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]

  return top_probs, top_labels, top_probs_idx

# Plot top_k predictions at a given time step
def plot_streaming_top_preds_at_step(
    top_probs,
    top_labels,
    step=None,
    image=None,
    legend_loc='lower left',
    duration_seconds=10,
    figure_height=500,
    playhead_scale=0.8,
    grid_alpha=0.3):
  """Generates a plot of the top video model predictions at a given time step.

  Args:
    top_probs: a tensor of shape (k, num_frames) representing the top-k
      probabilities over all frames.
    top_labels: a list of length k that represents the top-k label strings.
    step: the current time step in the range [0, num_frames].
    image: the image frame to display at the current time step.
    legend_loc: the placement location of the legend.
    duration_seconds: the total duration of the video.
    figure_height: the output figure height.
    playhead_scale: scale value for the playhead.
    grid_alpha: alpha value for the gridlines.

  Returns:
    A tuple of the output numpy image, figure, and axes.
  """
  import matplotlib as mpl
  import PIL
  # find number of top_k labels and frames in the video
  num_labels, num_frames = top_probs.shape
  if step is None:
    step = num_frames
  # Visualize frames and top_k probabilities of streaming video
  fig = plt.figure(figsize=(30, 15), dpi=150)
  gs = mpl.gridspec.GridSpec(8, 1)
  ax2 = plt.subplot(gs[:-3, :])
  ax = plt.subplot(gs[-3:, :])
  # display the frame
  if image is not None:
    ax2.imshow(image, interpolation='nearest')
    ax2.axis('off')
  # x-axis (frame number)
  preview_line_x = tf.linspace(0., duration_seconds, num_frames)
  # y-axis (top_k probabilities)
  preview_line_y = top_probs

  line_x = preview_line_x[:step+1]
  line_y = preview_line_y[:, :step+1]

  for i in range(num_labels):
    ax.plot(preview_line_x, preview_line_y[i], label=None, linewidth='1.5',
            linestyle=':', color='gray')
    ax.plot(line_x, line_y[i], label=top_labels[i], linewidth='2.0')


  ax.grid(which='major', linestyle=':', linewidth='1.0', alpha=grid_alpha)
  ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=grid_alpha)

  min_height = tf.reduce_min(top_probs) * playhead_scale
  max_height = tf.reduce_max(top_probs)
  ax.vlines(preview_line_x[step], min_height, max_height, colors='red')
  ax.scatter(preview_line_x[step], max_height, color='red')

  #ax.legend(loc=legend_loc)
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':15})
  plt.xlim(0, duration_seconds)
  plt.ylabel('Probability')
  plt.xlabel('Time (s)')
  plt.yscale('log')

  fig.tight_layout()
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  figure_width = int(figure_height * data.shape[1] / data.shape[0])
  image = PIL.Image.fromarray(data).resize([figure_width, figure_height])
  image = np.array(image)

  return image

# Plotting top_k predictions from MoViNets streaming model
def plot_streaming_top_preds(
    probs,
    video,
    labels,
    top_k=5,
    video_fps=25.,
    figure_height=500,
    use_progbar=True):
  """Generates a video plot of the top video model predictions.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    video: the video to display in the plot.
    top_k: the number of top predictions to select.
    video_fps: the input video fps.
    figure_fps: the output video fps.
    figure_height: the height of the output video.
    use_progbar: display a progress bar.

  Returns:
    A numpy array representing the output video.
  """
  # select number of frames per second
  video_fps = 8.
  # select height of the image
  figure_height = 500
  # number of time steps of the given video
  steps = video.shape[0]
  # estimate duration of the video (in seconds)
  duration = steps / video_fps
  # estiamte top_k probabilities and corresponding labels
  top_probs, top_labels, _ = get_top_k_streaming_labels(probs, labels, k=top_k)

  images = []
  step_generator = tqdm.trange(steps) if use_progbar else range(steps)
  for i in step_generator:
    image = plot_streaming_top_preds_at_step(
        top_probs=top_probs,
        top_labels=top_labels,
        step=i,
        image=video[i],
        duration_seconds=duration,
        figure_height=figure_height,
    )
    images.append(image)

  return np.array(images)

def save_to_video(frames):
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation.mp4', fourcc, 30.0, (frames.shape[2], frames.shape[1]))
    
    # Write each frame to the video file
    for frame in frames:
        out.write(frame)
    
    # Release the video writer and close the file
    out.release()
    
def AIPCreateBackboneAndClassifierModel(model_id, num_classes, frames_number, batch_size, resolution, 
                                        train_whole_model, dropout,
                                        checkpoint_dir,
                                        conv_type: str = '3d', se_type: str = '3d', activation: str = 'swish',
                                        gating_activation: str = 'sigmoid', stream_mode=False, input_specs=None, load_pretrained_weights=True, training = True):
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
                            #input_specs=input_specs,
                            activation=activation,
                            gating_activation=gating_activation,
                            #use_sync_bn=True,
                            #use_positional_encoding=True,
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
        #input_specs=input_specs,
        activation=activation,
        num_classes=num_classes, 
        output_states=stream_mode,
        dropout_rate = dropout
        )
  
  model.build([batch_size, frames_number, resolution, resolution, 3])
  return model
