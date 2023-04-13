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
        self.frame_step = frame_step
        self.output_signature = output_signature
    
    def train(self):
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(0.001)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file, 
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)
        
        train, val = Utils.remove_paths(self.train_ds), Utils.remove_paths(self.val_ds)
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_freq=1,
                            verbose=1)
        
        return results
    
    def test(self):
        test = Utils.remove_paths(self.test_ds)
        self.actual, self.predicted = Utils.get_actual_predicted_labels(test, self.base_model)
        self.test_acc = balanced_accuracy_score(self.actual, self.predicted)
        
    def predict(self, ds):
        labels = self.base_model.predict(ds)
        return labels
        
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
                                                                           frame_step=self.frame_step,
                                                                           shots=self.shots),
                                                             output_signature = self.output_signature)
            self.train_ds = train_ds.batch(self.batch_size)
        
        if val_path != "":
            val_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(val_path), self.num_frames,
                                                                           resolution = self.resolution,
                                                                           frame_step=self.frame_step,
                                                                           extension=extension),
                                                             output_signature = self.output_signature)
            self.val_ds = val_ds.batch(self.batch_size)
            
        if test_path != "":
            test_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(test_path), self.num_frames,
                                                                           resolution = self.resolution,
                                                                           frame_step=self.frame_step,
                                                                           extension=extension),
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
