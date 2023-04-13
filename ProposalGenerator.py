import Utils
import cv2
import numpy as np

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
    
    def sliding_frames_from_video(self, starting_frame, src, max_combined=3):
        
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
        label = np.argmax(self.model.predict(np_result[np.newaxis, :])[0])
        next_label = np.argmax(self.model.predict(np_next_result[np.newaxis, :])[0])
        
        counter = 0
        while (label == next_label) & (counter < max_combined):
            
            result = result+next_result.copy()
            result = result[::2]
            next_result = []
            counter+=1
            
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
            label = next_label #assume that lengthening the snippet does not chagne predicted label since label==next_label
            next_label = np.argmax(self.model.predict(np_next_result[np.newaxis, :])[0])
            
        stop_index = src.get(cv2.CAP_PROP_POS_FRAMES)
            
        #(processed_frames, start_index, stop_index)
        return np_result, int(starting_frame), int(stop_index)
    
    def __call__(self):
        
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.starting_frame = 0
        
        while self.starting_frame < (total_frames-self.n_frames):
            
            #should get this back from the function that processes sliding window
            processed_frames, start_index, stop_index = self.sliding_frames_from_video(self.starting_frame, cap)
            self.starting_frame = stop_index+1
            
            yield processed_frames, start_index, stop_index
