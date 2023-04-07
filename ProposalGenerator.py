import Utils
import cv2
import numpy as np
import tensorflow as tf

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
        print(starting_frame, stop_index)
        return np_result, int(starting_frame), int(stop_index)
    
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
            processed_frames, start_index, stop_index = self.sliding_frames_from_video(starting_frame, cap)
            starting_frame = stop_index+1
            
            print(start_index, stop_index)
            yield processed_frames, start_index, stop_index
            
if __name__ == '__main__':
    
    epochs_active_learning=1
    shots = -1
    drop_out = 0.5
    batch_size=16
    frame_step=6
    a_id='a2'
    loops=3
    num_samples=10
    
    model = ActiveLearningModel(
                num_loops=loops, num_samples=num_samples,
                model_id=a_id, model_type="base", 
                epochs=epochs_active_learning, shots=shots, 
                dropout=drop_out, 
                resolution=Utils.MOVINET_PARAMS[a_id][0], 
                num_frames=Utils.MOVINET_PARAMS[a_id][1]*5, 
                num_classes=12,
                batch_size=batch_size, 
                frame_step=frame_step,
                output_signature=Utils.OUTPUT_SIGNATURE,
                label_names=Utils.LABEL_NAMES)
    
    model.init_data('.mp4', "data/self/ercan", "data/self/roos", "data/self/joris")
    model.init_base_model()
    # try:
    #     model.load_best_weights()
    # except:
    #     print('Weights not found, training instead.')
    #     model.train()
    #     model.load_best_weights()
    
    pg = ProposalGenerator(model.base_model, "data/self/short.mp4", 
                           Utils.MOVINET_PARAMS[a_id][1]*5, Utils.MOVINET_PARAMS[a_id][0], 
                           model.frame_step)
    
    unlabeled_ds = tf.data.Dataset.from_generator(pg, output_signature=Utils.GENERATOR_SIGNATURE)
    unlabeled_ds = unlabeled_ds.batch(batch_size)
   
    model.unlabeled_ds = unlabeled_ds
    for vid, start, stop in model.unlabeled_ds.unbatch():
        print(vid.shape, start, stop)
    
    model.train()

    
    
    preds = model.predict(unlabeled_ds)
    print(np.argmax(preds, axis=1))
    
    # baseline.train_ds = train_ds
    # baseline.val_ds = train_ds
    # baseline.test_ds = cpy
    
    #baseline.test()
    #baseline.plot_confusion_matrix()
    
    
    #TODO
    #store actual frames in ds
    #allow extending/cutting of frames, store in ds and to file
    #also automatically and from and to previous instances. See how that works if continuously generated. 
    
    
    
    
    
    
    
    
    
    
    