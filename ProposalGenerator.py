import Utils
from Baseline import BaselineModel
import cv2
import numpy as np

UNLABELED_PATH = 'data/self/long.mp4'

model = BaselineModel(
            model_id='a0', model_type="base", 
            epochs=100, shots=5, 
            dropout=0.5, 
            resolution=172, 
            num_frames=32, 
            num_classes=12,
            batch_size=16, 
            output_signature=Utils.OUTPUT_SIGNATURE,
            label_names=Utils.LABEL_NAMES
)

model.init_data('.mp4', "data/self/joris", "data/self/ercan", "data/self/roos")
model.init_base_model()
try:
    model.load_best_weights()
except:
    print('Weights not found, training instead.')
    model.train()
    model.load_best_weights()
model.test()
model.plot_confusion_matrix()

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

class ProposalGenerator:
    def __init__(self, model, path, n_frames, resolution, frame_step, extension='.mp4'):

        self.model = model
        self.path = path
        self.n_frames = n_frames
        self.output_size = (resolution, resolution)
        self.extension = extension
        self.frame_step = frame_step
    
    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*'+self.extension))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes
    
    def sliding_frames_from_video(self, starting_frame, src):
        
        result = []
        next_result = []
        src.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        
        for _ in range(self.n_frames - 1):
            for _ in range(self.frame_step):
                ret, frame = src.read()
            if ret:
                frame = Utils.format_frames(frame, self.output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
                
        for _ in range(self.n_frames - 1):
            for _ in range(self.frame_step):
                ret, frame = src.read()
            if ret:
                frame = Utils.format_frames(frame, self.output_size)
                next_result.append(frame)
            else:
                next_result.append(np.zeros_like(result[0]))
        
        np_result = np.array(result)[..., [2, 1, 0]]
        np_next_result = np.array(next_result)[..., [2, 1, 0]]
        label, next_label = self.model.predict([np_result, np_next_result])
        
        while label == next_label:
            
            result = result+next_result.copy()
            result = result[::2]
            next_result = []
            
            for _ in range(self.n_frames - 1):
                for _ in range(self.frame_step):
                    ret, frame = src.read()
                if ret:
                    frame = Utils.format_frames(frame, self.output_size)
                    next_result.append(frame)
                else:
                    next_result.append(np.zeros_like(result[0]))
            
            np_result = np.array(result)[..., [2, 1, 0]]
            np_next_result = np.array(next_result)[..., [2, 1, 0]]
            label, next_label = self.model.predict([result, next_result])

        stop_index = src.get(cv2.CAP_PROP_POS_FRAMES)
            
        #(processed_frames, start_index, stop_index)
        return np_result, starting_frame, stop_index
    
    def __call__(self):
        
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        #e.g. fps=15, total_frames=1822, num_frames=32 then every snippet is
        #about 2 seconds. 
        n_snippets = int(total_frames/self.n_frames)
        
        starting_frame = 0
        while starting_frame < (total_frames-self.n_frames):
            
            #should get this back from the function that processes sliding window
            processed_frames, start_index, stop_index = self.sliding_frames_from_video(starting_frame, capst)
            starting_frame = stop_index+1
            
            yield processed_frames, start_index, stop_index
        
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        
        for path, name in pairs:
            video_frames = Utils.frames_from_video_file(path, self.n_frames, output_size=(self.resolution, self.resolution)) 


            yield processed_frames, start_index, stop_index