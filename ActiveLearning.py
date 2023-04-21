import os
import datetime
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Union

import Utils
from Baseline import BaselineModel


class ActiveLearningModel(BaselineModel):
    
    unlabeled_ds: tf.data.Dataset
    labeled_ds: tf.data.Dataset
    
    num_loops: int
    num_samples: int
    
    def __init__(self, num_loops, data_path, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AL"
        self.weights_file = f'/movinet_{self.name}_{self.model_id}_{self.model_type}_weights.hdf5'
        self.num_loops = num_loops
        self.num_samples = num_samples
        self.unlabeled_paths = []
        self.unlabeled_path = ""
        os.makedirs(Utils.AL_FOLDER, exist_ok=True)
        
        # Read the INCLUDE file and create a set of video codes to be included
        with open('data/slapi/INCLUDE', 'r') as f:
            included_video_codes = set(line.strip() for line in f.readlines())
        
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp4"):
                    self.unlabeled_paths.append(os.path.join(root, file))
        self.unlabeled_paths = [path for path in self.unlabeled_paths if not any(video_code in path for video_code in included_video_codes)]
    
    def init_data(self, *args, **kwargs):
        super().init_data(*args, **kwargs)
        self.labeled_ds = self.train_ds
    
    def init_unlabeled_data(self, path, extension='.mp4'):
            unlabeled_ds = tf.data.Dataset.from_generator(Utils.ProposalGenerator(self.base_model, path, 
                                   self.num_frames, Utils.MOVINET_PARAMS[self.model_id][0], 
                                   self.frame_step), output_signature = Utils.GENERATOR_SIGNATURE)
            self.unlabeled_ds = unlabeled_ds.batch(self.batch_size)

    
    def get_labels(self, path, start_indices, stop_indices, samples):
        labels, return_samples, saved_paths = [], [], []
        if not os.path.exists(Utils.AL_FOLDER): os.mkdir(Utils.AL_FOLDER)
        
        for i in range(len(start_indices)):
            start, stop = start_indices[i], stop_indices[i]
            
            def print_instructions():
                os.system('cls' if os.name == 'nt' else 'clear')
                print("LABELS")
                for i in range(len(self.label_names)): print(f'{i}: {self.label_names[i]}')
                print('s: SKIP snippet\nr: REPLAY snippet\nm: MODIFY start/stop\no: OTHER class label (please enter)')
            
            def play_video(start: int, stop: int) -> List[np.ndarray]:
                cap, frames, fps, counter = cv2.VideoCapture(path), [], 30, 0
                cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                
                while cap.get(cv2.CAP_PROP_POS_FRAMES) < stop:
                    counter += 1
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                    show_frame = cv2.putText(frame.copy(), f"{int(counter/fps)}/{int((stop-start)/fps)} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Video", show_frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key in {ord('r'), ord('R')}: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    elif key in {ord('p'), ord('P')}: cv2.waitKey(0)
                
                cap.release()
                cv2.destroyAllWindows()
                
                return frames
            
            def save_video(played_frames: List[np.ndarray], label: Union[int, str]) -> None:
                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S").replace(":", "_")
                name = f"{time_str}-{self.unlabeled_path[-7:-4]}"
                vid_dir = f'{Utils.AL_FOLDER}/{label if isinstance(label, str) else Utils.LABEL_NAMES[label]}'
                name = f'{label if isinstance(label, str) else Utils.LABEL_NAMES[label]}-{name}'
                
                if isinstance(label, int): saved_paths.append(f"{vid_dir}/{name}.mp4")
                if not os.path.exists(vid_dir): os.mkdir(vid_dir)
                out = cv2.VideoWriter(f"{vid_dir}/{name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (played_frames[0].shape[1], played_frames[0].shape[0]))
                for frame in played_frames: out.write(frame)
                out.release()
            
            def confirm_label(label: int, played_frames: List[np.ndarray]) -> bool:
                if label not in range(len(self.label_names)):
                    print("Invalid input, please enter a label between 0 and", len(self.label_names)-1)
                    return False
                
                labels.append(tf.cast(label, tf.int32))
                return_samples.append(samples[i]) 
                save_video(played_frames, label)     
                
                return True
            
            def change_video_length(change: str = "") -> Tuple[int, int, bool]:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Do you want to add or remove frames to the front or back\nFirst enter how many seconds you want to add/remove from the front\nThen enter how many seconds you want to add/remove to the back\n'-0.5' removes half a second, 2 adds two seconds\nFor example,-0.5:2 to remove 0.5 seconds from front and add 2 seconds to the end.")
                
                if not change: change = input("Enter in format [s:s], 'q' to skip: ")
                if change.lower() == 'q': return start, stop, False
                else:
                    try: change_start, change_end = change.split(':'); change_start, change_end = int(float(change_start)*30), int(float(change_end)*30)
                    except: print("Did not recognize input, try again"); return change_video_length()
                
                return start + change_start, change_end + stop, True
            
            def create_new_sample(start: int, stop: int) -> np.ndarray:
                result, cap = [], cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                
                for _ in range(self.num_frames-1):
                    for _ in range(self.frame_step): ret, frame = cap.read()
                    if ret: frame = Utils.format_frames(frame, (self.resolution, self.resolution)); result.append(frame)
                    else: result.append(np.zeros_like(result[0]))
                cap.release() 
                
                return np.array(result)[..., [2, 1, 0]]
                
            
            label = None
            while label is None:
                try:
                    played_frames = play_video(start, stop); print_instructions(); label_str = input("Please select label (number): ")
                
                    if label_str.lower() == 'r': continue
                    if label_str.lower() == 'm': start, stop, changed = change_video_length(); samples[i] = create_new_sample(start, stop) if changed else samples[i]; continue
                    if ":" in label_str: start, stop, changed = change_video_length(label_str); samples[i] = create_new_sample(start, stop) if changed else samples[i]; continue
                    if label_str.lower() == 'o': other_label = input("Please enter class label in text (a-z): "); save_video(played_frames, other_label); break
                    if label_str.lower() == 's': break
                    
                    try: 
                        label = int(label_str)
                        if confirm_label(label, played_frames): break
                    except Exception as e: print(f"This gave error {e}, please enter a valid label number")
                
                except Exception as ex: print(f"Something went wrong with this label, skipping it: {ex}"); break
        
        return np.array(labels), np.array(return_samples), np.array(saved_paths)

    def select_samples(self, labeled_ds, unlabeled_ds, num_samples):

        data = [(v, start, stop) for (v, start, stop) in unlabeled_ds.unbatch()]
        vids = np.array([v for v, _, _ in data])
        
        # Highest entropy score across logits
        def uncertainty_sampling(vids, num_samples):
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            entropies = -tf.reduce_sum(unlabeled_probs * tf.math.log(unlabeled_probs), axis=-1)
            indices = tf.argsort(entropies, direction='DESCENDING')
            return indices[:num_samples]
        
        #Most diverse in terms of feature representation
        def diversity_sampling(vids, num_samples, k=5):
            features = self.base_model.backbone(vids)
            features = np.array([f.numpy().flatten() for f in features[0]['head']])
            distance_matrix = pairwise_distances(features)
            diversity_scores = np.mean(np.sort(distance_matrix)[:, 1:k+1], axis=1)
            return np.argsort(diversity_scores)[-num_samples:]
        
        # Minimum predicted probability for predicted class
        def least_confident_sampling(vids, num_samples):
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            min_probs = tf.reduce_min(unlabeled_probs, axis=-1)
            indices = tf.argsort(min_probs, direction='DESCENDING')
            return indices[:num_samples]

        # BvSB
        def margin_sampling(vids, num_samples):
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            sorted_probs = tf.sort(unlabeled_probs, axis=-1, direction='DESCENDING')
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            indices = tf.argsort(margin, direction='ASCENDING')
            return indices[:num_samples]
        
        # More certain samples over time
        def dynamic_sampling(vids, num_samples, iteration, max_iterations):
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            entropies = -tf.reduce_sum(unlabeled_probs * tf.math.log(unlabeled_probs), axis=-1)
            threshold = 1 - (iteration / max_iterations)
            selected_indices = tf.where(entropies < threshold)
            return tf.squeeze(selected_indices)[:num_samples]
        
        #selects indices per class max activation and corrected per class        
        def select_weighted_top_samples(vids, classes, samples_per_class):
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            
            top_indices = {}
            for class_idx in range(len(classes)):
                class_probs = unlabeled_probs[:, class_idx]
                topk_indices = tf.math.top_k(class_probs, k=samples_per_class[class_idx]).indices
                top_indices[class_idx] = topk_indices.numpy()
            
            selected_indices = np.unique(np.concatenate(list(top_indices.values())))
            return selected_indices


    
        selected_indices = select_weighted_top_samples(vids, Utils.LABEL_NAMES, Utils.get_class_weights(self.unlabeled_ds))
        processed_frames, start_indices, stop_indices = zip(*data)
        processed_frames, start_indices, stop_indices = np.array(processed_frames), np.array(start_indices), np.array(stop_indices)
    
        selected_samples = [processed_frames[i] for i in selected_indices]
        selected_start = [start_indices[i] for i in selected_indices]
        selected_stop = [stop_indices[i] for i in selected_indices]
    
        selected_labels, selected_samples, selected_paths = self.get_labels(self.unlabeled_path, selected_start, selected_stop, selected_samples)
    
        labeled_ds = labeled_ds.unbatch().concatenate(tf.data.Dataset.from_tensor_slices((selected_samples, selected_labels, selected_paths))).shuffle(buffer_size=len(vids)+len(selected_samples)).batch(self.batch_size)
    
        remaining_indices = np.delete(np.arange(len(processed_frames)), selected_indices)
        unlabeled_ds = tf.data.Dataset.from_tensor_slices((processed_frames[remaining_indices], start_indices[remaining_indices], stop_indices[remaining_indices])).batch(self.batch_size)
    
        return labeled_ds, unlabeled_ds
    
    def train(self):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir + self.weights_file,
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

        performance_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for i in range(self.num_loops):
            train, val = Utils.remove_paths(self.labeled_ds), Utils.remove_paths(self.val_ds)

            results = self.base_model.fit(train,
                                validation_data=val,
                                epochs=self.epochs,
                                callbacks=[model_checkpoint, early_stopping],
                                validation_freq=1,
                                class_weight=Utils.get_class_weights(train),
                                verbose=1)

            for key in results.history.keys():
                performance_history[key].extend(results.history[key])
                
            os.system('cls' if os.name == 'nt' else 'clear')

            print(f'Starting Active Learning loop {i+1}/{self.num_loops}')
            tf.keras.backend.clear_session()
            
            #Maybe not do this every time, but just recompute uncertainty over self.unlabeled_ds since it is still available
            #maybe only do this as fucntion of num_loops/len(self.available paths)
            #self.unlabeled_path = self.unlabeled_paths.pop()
            #self.init_unlabeled_data(self.unlabeled_path)
            
            self.labeled_ds, self.unlabeled_ds = self.select_samples(self.labeled_ds, self.unlabeled_ds, self.num_samples)
            
            os.system('cls' if os.name == 'nt' else 'clear')
    
        train, val = Utils.remove_paths(self.labeled_ds), Utils.remove_paths(self.val_ds)
        # Train the final model using the best weights
        self.base_model.load_weights(self.checkpoint_dir + self.weights_file)
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_freq=1,
                            class_weight=Utils.get_class_weights(train),
                            verbose=1)

        for key in results.history.keys():
            performance_history[key].extend(results.history[key])
        
        self.history = performance_history

