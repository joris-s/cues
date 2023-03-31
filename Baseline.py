import tensorflow as tf
import Utils
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

class BaselineModel:
    
    name:str
    model_id: str
    model_type: str
    base_model: tf.keras.Model
    stream_model: tf.keras.Model
    
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    label_names: list
    
    resolution: int
    batch_size: int
    dropout: int
    num_frames: int
    num_classes: int
    shots: int
    extension: str
    checkpoint_dir: str
    weights_file:str
    epochs: int
    output_signature: list
    
    test_acc: float
    
    def __init__(self, model_id, model_type, epochs, shots, dropout,
                 resolution, num_frames, num_classes, label_names, batch_size, 
                 frame_step, output_signature):
        self.name = "Baseline"
        self.model_id = model_id
        self.model_type = model_type
        self.checkpoint_dir = f'MoViNets/movinet_{model_id}_{model_type}'
        self.weights_file = f'/movinet_{self.name}_{model_id}_{model_type}_weights.hdf5'
        self.epochs = epochs
        self.resolution = resolution
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.label_names = label_names
        self.shots = shots
        self.dropout = dropout
        self.batch_size = batch_size
        self.frame_step = frame_step,
        self.output_signature = output_signature
    
    def train(self):
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(0.001)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file, 
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)
        
        train, val = Utils.remove_paths(self.train_ds, self.val_ds)
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_freq=1,
                            verbose=1)
        
        return results
    
    def test(self):
        _, test = Utils.remove_paths(self.train_ds, self.test_ds)
        self.actual, self.predicted = Utils.get_actual_predicted_labels(test, self.base_model)
        self.test_acc = balanced_accuracy_score(self.actual, self.predicted)
        
    def load_best_weights(self):
        self.base_model.load_weights(self.checkpoint_dir+self.weights_file)
    
    def generate_segment(self):
        pass
    
    def init_base_model(self):
        self.base_model = Utils.AIPCreateBackboneAndClassifierModel(model_id=self.model_id, 
                                                         num_classes=self.num_classes, 
                                                         frames_number=self.num_frames, 
                                                         batch_size=self.batch_size, 
                                                         resolution=self.resolution, 
                                                         train_whole_model=False, 
                                                         dropout=self.dropout,
                                                         checkpoint_dir=self.checkpoint_dir,
                                                         stream_mode=False)
    
    def init_streaming_model(self):
        self.stream_model = Utils.AIPCreateBackboneAndClassifierModel(model_id=self.model_id, 
                                                         num_classes=self.num_classes, 
                                                         frames_number=self.num_frames, 
                                                         batch_size=self.batch_size, 
                                                         resolution=self.resolution, 
                                                         train_whole_model=False, 
                                                         dropout=0.0,
                                                         checkpoint_dir=self.checkpoint_dir,
                                                         stream_mode=True)

        self.stream_model.set_weights(self.base_model.get_weights())
    
    def init_data(self, extension, train_path = "", val_path = "", test_path = ""):
        if train_path != "":
            train_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(train_path), self.num_frames,
                                                                           resolution = self.resolution,
                                                                           training = True, 
                                                                           extension=extension,
                                                                           shots=self.shots,
                                                                           frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.train_ds = train_ds.batch(self.batch_size)
        
        if val_path != "":
            val_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(val_path), self.num_frames, 
                                                                           resolution = self.resolution,
                                                                           extension=extension,
                                                                           frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.val_ds = val_ds.batch(self.batch_size)
            
        if test_path != "":
            test_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(test_path), self.num_frames,
                                                                           resolution = self.resolution,
                                                                           extension=extension,
                                                                           frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.test_ds = test_ds.batch(self.batch_size)
            
    def plot_confusion_matrix(self, savefig=True):
        try:
            Utils.cm_heatmap(self.actual, self.predicted, self.label_names, savefig, 
                       (f'Confusion Matrix {self.name} for {self.model_id} - %.2f Acc' % self.test_acc))
        except Exception as e:
            print(f"We found the error: {e}\nNow doing testing again...")
            self.actual, self.predicted = Utils.get_actual_predicted_labels(self.test_ds, self.base_model)
            Utils.cm_heatmap(self.actual, self.predicted, self.label_names, savefig, 
                       (f'Confusion Matrix {self.name} for {self.model_id} - %.2f Acc' % self.test_acc))

if __name__ == '__main__':

    
    """-----------------------------------"""
    """         STREAMING                 """
    """-----------------------------------"""
    
# baseline.init_streaming_model()
# video = Utils.load_video('data/self/long/long.mp4')
# images = tf.split(video[tf.newaxis], video.shape[0], axis=1)
# images = images[0::15]
# init_states = baseline.stream_model.init_states(video[tf.newaxis].shape)

# all_logits = []

# # # To run on a video, pass in one frame at a time
# states = init_states
# for i in range(len(images)):
#      print(f"{i}/{len(images)}")
#      # predictions for each frame
#      image = images[i]
#      logits, states = baseline.stream_model({**states, 'image': image})
#      #logits = baseline.base_model({'image': image})
#      all_logits.append(logits)

# # concatinating all the logits
# logits = tf.concat(all_logits, 0)
# # estimating probabilities
# probs = tf.nn.softmax(logits, axis=-1)
# labels = baseline.label_names


# # Generate a plot and output to a video tensor
# plot_video = Utils.plot_streaming_top_preds(np.repeat(probs, 15, axis=0), video, labels=labels, video_fps=15., top_k=12)
# # For gif format, set codec='gif'
# Utils.save_to_video(plot_video)

# import cv2
# cap = cv2.VideoCapture('data/self/long/long.mp4')
# num_frames = 10
# step = 5
# class_labels = ['frown', 'frown', 'frown', 'frown', 'frown', 'frown', 'frown', 'frown', 'frown', 'frown']
# # Loop over the frames from the video capture device
# while True:
#     # Read a frame from the video capture device
#     frames = []    
#     ret, frame = cap.read()

#     # If the frame was not successfully read, break out of the loop
#     if not ret:
#         break
#     # Preprocess the frame
    
#     for _ in range(num_frames):
#         for _ in range(step):
#             ret, frame = cap.read()
#         if ret:
#           input_frame = Utils.preprocess_frame(frame)
#           frames.append(input_frame)
          
#     input_frames = np.stack(frames, axis = 1)
 
#     # Make a prediction with the model
#     prediction = baseline.base_model.predict(input_frames)
#     # Get the class label with the highest probability
#     pred = np.argmax(prediction)
#     class_labels.append(labels[pred])
#     class_labels = class_labels[1:5]
#     class_label = max(set(class_labels), key = class_labels.count)
#     # Display the class label on the frame
#     cv2.putText(frame, str(class_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#     # Display the frame
#     cv2.imshow('Video Feed', frame)
#     # If the 'q' key is pressed, break out of the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release
# cap.release()
# cv2.destroyAllWindows()