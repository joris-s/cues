import tensorflow as tf
import Utils
import numpy as np
import cv2
from Baseline import BaselineModel
import os


class ActiveLearningModel(BaselineModel):
    
    unlabeled_ds: tf.data.Dataset
    labeled_ds: tf.data.Dataset
    
    num_loops: int
    num_samples: int
    
    def __init__(self, num_loops, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AL"
        self.weights_file = f'/movinet_{self.name}_{self.model_id}_{self.model_type}_weights.hdf5'
        self.num_loops = num_loops
        self.num_samples = num_samples
    
    def init_data(self, *args, **kwargs):
        super().init_data(*args, **kwargs)
        self.unlabeled_ds = self.val_ds
        self.labeled_ds = self.train_ds
    
    def get_labels(self, paths):
        labels = []
        
        for path in paths:
            path = path.decode('utf-8')
            
            #Print the labels and instructions
            def print_instructions():
                os.system('cls' if os.name == 'nt' else 'clear')
                print("LABELS")
                for i in range(len(self.label_names)):
                    print(f'{i}: {self.label_names[i]}')
                print("r: REPLAY")
            
            #Play the video
            def play_video():
                
                # Load the video using OpenCV
                cap = cv2.VideoCapture(path)
                
                # Create a window to display the video
                cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
                #cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                #cv2.resizeWindow("Video", 1280, 720)
                #cv2.moveWindow("Video", 100, 100)
                
                while True:
                    
                    # Read a frame from the video
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow("Video", frame)
                    
                    # Handle key events
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('r') or key == ord('R'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if key == ord('p') or key == ord('P'):
                        cv2.waitKey(0)
                        
                cap.release()
                cv2.destroyAllWindows()
                
            # Define a function to confirm the label and move on to the next video
            def confirm_label(label):
                if label not in range(len(self.label_names)):
                    print("Invalid input, please enter a label between 0 and", len(self.label_names)-1)
                    return
                nonlocal labels
                labels.append(label)
                
            label = None    
            while label == None:
                
                play_video()
                print_instructions()
                label_str = input("Please select label (number): ")
                if label_str == 'r' or label_str == 'R':
                    continue
                
                try: 
                    label = int(label_str)
                    confirm_label(label)
                except Exception as e:
                    print(f"This gave error {e}, please enter a valid label number")
                    
        return labels                    
                   
    
    def select_samples(self, labeled_ds, unlabeled_ds, num_samples):
        # Get predicted probabilities for unlabeled samples
        unlabeled_probs = tf.nn.softmax(self.base_model.predict(unlabeled_ds))
    
        # Get maximum entropy for each sample
        entropies = tf.math.log(unlabeled_probs) * unlabeled_probs
        entropies = tf.reduce_sum(entropies, axis=-1)
    
        # Select samples with highest entropy
        indices = tf.argsort(entropies, direction='ASCENDING')
        selected_indices = indices[:num_samples]
        selected_indices = tf.cast(indices[:num_samples], dtype=tf.int64)  # cast to int64
        
        # Convert dataset to numpy iterator
        iterator = unlabeled_ds.unbatch().as_numpy_iterator()
    
        # Iterate over the dataset to extract frames and labels as numpy arrays
        frames = []
        labels = []
        paths = []
        for example in iterator:
            frames.append(example[0])
            labels.append(example[1])
            paths.append(example[2])
    
        # Convert lists to numpy arrays
        frames = np.array(frames)
        labels = np.array(labels)
        paths = np.array(paths)
    
        # Get corresponding samples from unlabeled dataset
        selected_samples = []
        selected_labels = []
        selected_paths = []
        for index in selected_indices:
            selected_sample = frames[index]
            selected_samples.append(selected_sample)
            selected_label = labels[index]
            selected_labels.append(selected_label)
            selected_path = paths[index]
            selected_paths.append(selected_path)
            
        frames = np.delete(frames, selected_indices, axis=0)
        labels = np.delete(labels, selected_indices)
        paths = np.delete(paths, selected_indices)
        
        selected_samples = np.array(selected_samples)
        selected_labels = np.array(selected_labels)
        selected_labels = np.array(self.get_labels(selected_paths))
        #selected_paths = np.array(selected_paths)
        
        unlabeled_ds = tf.data.Dataset.from_tensor_slices((frames, labels, paths))
        unlabeled_ds = unlabeled_ds.batch(self.batch_size)
        
        selected_frames_tensor = tf.constant(selected_samples)
        selected_labels_tensor = tf.convert_to_tensor(selected_labels, dtype=tf.int16)
        selected_paths_tensor = tf.constant(selected_paths)
        labeled_ds = labeled_ds.unbatch().concatenate(tf.data.Dataset.from_tensor_slices((selected_frames_tensor, selected_labels_tensor, selected_paths_tensor)))
        labeled_ds = labeled_ds.batch(self.batch_size)
    
        return labeled_ds, unlabeled_ds
    
    def train(self):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0001)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file,
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(0.001)

        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

        train, val = Utils.remove_paths(self.labeled_ds, self.unlabeled_ds)
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping, reduce_lr],
                            validation_freq=1,
                            verbose=1)

        for i in range(self.num_loops):
            os.system('cls' if os.name == 'nt' else 'clear')

            print(f'Starting Active Learning loop {i+1}/{self.num_loops}')
            tf.keras.backend.clear_session()
    
            self.labeled_ds, self.unlabeled_ds = self.select_samples(self.labeled_ds, self.unlabeled_ds, self.num_samples)
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f'Continuing training loop {i+1}/{self.num_loops}')

            train, val = Utils.remove_paths(self.labeled_ds, self.unlabeled_ds)
            self.base_model.optimizer.lr = 0.001
            results = self.base_model.fit(train,
                                validation_data=val,
                                epochs=self.epochs,
                                callbacks=[model_checkpoint, early_stopping, reduce_lr],
                                validation_freq=1,
                                verbose=1)

        self.base_model.optimizer.lr = 0.001
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping, reduce_lr],
                            validation_freq=1,
                            verbose=1)
        
        return results

