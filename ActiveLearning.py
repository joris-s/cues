import tensorflow as tf
import Utils
import numpy as np
import cv2
from Baseline import BaselineModel
import os
from ProposalGenerator import ProposalGenerator
import datetime
from sklearn.metrics import pairwise_distances


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
        
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp4"):
                    self.unlabeled_paths.append(os.path.join(root, file))
    
    def init_data(self, *args, **kwargs):
        super().init_data(*args, **kwargs)
        self.labeled_ds = self.train_ds
    
    def init_unlabeled_data(self, path, extension='.mp4'):
            unlabeled_ds = tf.data.Dataset.from_generator(ProposalGenerator(self.base_model, path, 
                                   self.num_frames, Utils.MOVINET_PARAMS[self.model_id][0], 
                                   self.frame_step), output_signature = Utils.GENERATOR_SIGNATURE)
            self.unlabeled_ds = unlabeled_ds.batch(self.batch_size)

    
    def get_labels(self, path, start_indices, stop_indices, samples):
        labels = []
        return_samples = []
        
        if not os.path.exists(Utils.AL_FOLDER):
            os.mkdir(Utils.AL_FOLDER)

        for i in range(len(start_indices)):
        
            start, stop = start_indices[i], stop_indices[i]
            
            #Print the labels and instructions
            def print_instructions():
                os.system('cls' if os.name == 'nt' else 'clear')
                print("LABELS")
                for i in range(len(self.label_names)):
                    print(f'{i}: {self.label_names[i]}')
                print('s: SKIP snippet')
                print("r: REPLAY snippet")
                print("m: MODIFY start/stop")
                print("o: OTHER class label (please enter)")
            
            #Play the video
            def play_video(start, stop):
                
                # Load the video using OpenCV
                cap = cv2.VideoCapture(path)
                cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                frames = []
                fps=30
                counter=0
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                while cap.get(cv2.CAP_PROP_POS_FRAMES) < stop:
                    
                    # Read a frame from the video
                    counter+=1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    show_frame = cv2.putText(frame.copy(), f"{int(counter/fps)}/{int((stop-start)/fps)} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Video", show_frame)
                    
                    # Handle key events
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('r') or key == ord('R'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if key == ord('p') or key == ord('P'):
                        cv2.waitKey(0)
                        
                cap.release()
                cv2.destroyAllWindows()
                
                return frames
            
            #Save the video
            def save_video(played_frames, label):
                now=datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S").replace(":", "_")
                name = f"{time_str}-{self.unlabeled_path[-7:-4]}"

                if type(label) == str:
                    vid_dir = f'{Utils.AL_FOLDER}/{label}'
                    name = f'{label}-{name}'
                else:
                    vid_dir = f'{Utils.AL_FOLDER}/{Utils.LABEL_NAMES[label]}'
                    name = f'{Utils.LABEL_NAMES[label]}-{name}'
                    
                if not os.path.exists(vid_dir):
                    os.mkdir(vid_dir)

                out = cv2.VideoWriter(f"{vid_dir}/{name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (played_frames[0].shape[1], played_frames[0].shape[0]))
                for frame in played_frames:
                    out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
                out.release()
                
            # Define a function to confirm the label and move on to the next video
            def confirm_label(label, played_frames):
                if label not in range(len(self.label_names)):
                    print("Invalid input, please enter a label between 0 and", len(self.label_names)-1)
                    return
                nonlocal labels
                labels.append(label)
                return_samples.append(samples[i])
                save_video(played_frames, label)
                
            def change_video_length(change = ""):
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Do you want to add or remove frames to the front or back")
                print("First enter how many seconds you want to add/remove from the front")
                print("Then enter how many seconds you want to add/remove to the back")
                print("'-0.5' removes half a second, 2 adds two seconds")
                print("For example,-0.5:2 to remove 0.5 seconds from front and add 2 seconds to the end.")
                
                if change == "":
                    change = input("Enter in format [s:s], 'q' to skip: ")
                
                if (change == 'q') or (change == 'Q'):
                    return start, stop, False
                else:
                    try:
                        change_start, change_end = change.split(':')
                        change_start, change_end = int(change_start)*30, int(change_end)*30
                    except:
                        print("Did not recognise input, try again")
                        return change_video_length()
                    
                return start+change_start, change_end+stop, True
            
            # Create a new sample after cutting/adding frames            
            def create_new_sample(start, stop):
                result = []
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                
                for _ in range(self.num_frames-1):
                  for _ in range(self.frame_step):
                    ret, frame = cap.read()
                  if ret:
                    frame = Utils.format_frames(frame, (self.resolution, self.resolution))
                    result.append(frame)
                  else:
                    result.append(np.zeros_like(result[0]))

                cap.release()
                result = np.array(result)[..., [2, 1, 0]]

                return result
                
            label = None    
            while label == None:
                try:
                    played_frames = play_video(start, stop)
                                
                    print_instructions()
                    label_str = input("Please select label (number): ")
                    if label_str == 'r' or label_str == 'R':
                        continue
                    
                    if label_str == 'm' or label_str == 'M':
                        start, stop, changed = change_video_length()
                        if changed:
                            samples[i] = create_new_sample(start, stop)
                            continue
                        
                    if ":" in label_str:
                        try:
                            start, stop, changed = change_video_length(label_str)
                            if changed:
                                samples[i] = create_new_sample(start, stop)
                                continue
                        except:
                            print("Not correct input for changing length, trying again")
                            continue
                    
                    if label_str == 'o' or label_str == 'O':
                        other_label = input("Please enter class label in text (a-z): ")
                        save_video(played_frames, other_label)
                        break
                    
                    if label_str == 's' or label_str == 'S':
                        break
                
                    try: 
                        label = int(label_str)
                        confirm_label(label, played_frames)
                    except Exception as e:
                        print(f"This gave error {e}, please enter a valid label number")
                
                except Exception as ex:
                    print(f"Something went wrong with this label, skipping it: {ex}")
                    break
        
        return np.array(labels), np.array(return_samples)                   
    
    def select_samples(self, labeled_ds, unlabeled_ds, num_samples):

        data = [d for d in unlabeled_ds.unbatch()]
        vids = np.array([v for v, _, _ in data])
        starts = [start for _, start, _ in data]
        stops = [stop for _, _, stop in data]
    
        def uncertainty_sampling(vids, num_samples):
                
            unlabeled_probs = tf.nn.softmax(self.base_model(vids))
            # Get maximum entropy for each sample
            entropies = tf.math.log(unlabeled_probs) * unlabeled_probs
            entropies = tf.reduce_sum(entropies, axis=-1)
            # Select samples with highest entropy
            indices = tf.argsort(entropies, direction='ASCENDING')
            return indices[:num_samples]
        
        def diversity_sampling(vids, num_samples, k=5):
            
            features = self.base_model.backbone(vids)
            features = np.array([np.array(f).flatten() for f in features[0]['head']])
            
            distance_matrix = pairwise_distances(features)
            diversity_scores = np.mean(np.sort(distance_matrix)[:, 1:k+1], axis=1)
            return np.argsort(diversity_scores)[-num_samples:]
            
        selected_indices = uncertainty_sampling(vids, num_samples)
        #selected_indices = diversity_sampling(vids, num_samples)

        # Iterate over the dataset to extract frames and labels as numpy arrays
        processed_frames = []
        start_indices = []
        stop_indices = []

        for i in range(len(vids)):
            example = (vids[i], starts[i], stops[i])
            processed_frames.append(example[0])
            start_indices.append(example[1])
            stop_indices.append(example[2])
        
        # Convert lists to numpy arrays
        processed_frames = np.array(processed_frames)
        start_indices = np.array(start_indices)
        stop_indices = np.array(stop_indices)
    
        # Get corresponding samples from unlabeled dataset
        selected_samples = []
        selected_start = []
        selected_stop = []
        
        for index in selected_indices:
            selected_samples.append(processed_frames[index])
            selected_start.append(start_indices[index])
            selected_stop.append(stop_indices[index])
            
        processed_frames = np.delete(processed_frames, selected_indices, axis=0)
        start_indices = np.delete(start_indices, selected_indices)
        stop_indices = np.delete(stop_indices, selected_indices)
        
        selected_labels, selected_samples = self.get_labels(self.unlabeled_path, selected_start, selected_stop, selected_samples)
        unlabeled_ds = tf.data.Dataset.from_tensor_slices((processed_frames, start_indices, stop_indices))
        unlabeled_ds = unlabeled_ds.batch(self.batch_size)
        
        selected_frames_tensor = tf.constant(selected_samples)
        selected_labels_tensor = tf.convert_to_tensor(selected_labels, dtype=tf.int32)
        selected_paths_tensor = tf.constant(np.array([f'class_{label}.mp4' for label in selected_labels]))
        
        labeled_ds = labeled_ds.unbatch().concatenate(tf.data.Dataset.from_tensor_slices((selected_frames_tensor, selected_labels_tensor, selected_paths_tensor)))
        labeled_ds = labeled_ds.shuffle(buffer_size=len(vids)+len(selected_samples)).batch(self.batch_size)
        return labeled_ds, unlabeled_ds
    
    def train(self):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0001)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file,
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(0.001)
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

        for i in range(self.num_loops):

            train, val = Utils.remove_paths(self.labeled_ds), Utils.remove_paths(self.val_ds)
            self.base_model.optimizer.lr = 0.001
            results = self.base_model.fit(train,
                                validation_data=val,
                                epochs=self.epochs,
                                callbacks=[model_checkpoint, early_stopping, reduce_lr],
                                validation_freq=1,
                                verbose=1)

            os.system('cls' if os.name == 'nt' else 'clear')

            print(f'Starting Active Learning loop {i+1}/{self.num_loops}')
            tf.keras.backend.clear_session()
            self.unlabeled_path = self.unlabeled_paths.pop()
            self.init_unlabeled_data(self.unlabeled_path)
            self.labeled_ds, self.unlabeled_ds = self.select_samples(self.labeled_ds, self.unlabeled_ds, self.num_samples)
            
            os.system('cls' if os.name == 'nt' else 'clear')
            

        self.base_model.optimizer.lr = 0.001
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping, reduce_lr],
                            validation_freq=1,
                            verbose=1)
        
        return results
