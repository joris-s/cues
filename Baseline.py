import tensorflow as tf
import Utils
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import json
import numpy as np


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
    
    history: dict
    
    def __init__(self, model_id, model_type, shots, dropout, resolution, 
                 num_frames, num_classes, label_names, version, batch_size, 
                 frame_step, train_backbone, regularization, output_signature):
        self.name = "Baseline"
        self.model_id = model_id
        self.model_type = model_type
        self.checkpoint_dir = f'MoViNets/movinet_{model_id}_{model_type}'
        self.weights_file = f'/movinet_{self.name}_{model_id}_{model_type}_weights{version}.hdf5'
        self.resolution = resolution
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.label_names = label_names
        self.version = version
        self.shots = shots
        self.dropout = dropout
        self.batch_size = batch_size
        self.frame_step = frame_step
        self.train_backbone = train_backbone
        self.regularization = regularization
        self.output_signature = output_signature
    
    def train(self, learning_rate=1e-3, epochs=5):
        ph = {m.name: [] for m in Utils.METRICS}
        performance_history = {'loss': [], 'val_loss': []}
        
        for metric_name in ph.keys():
            performance_history[f'train_{metric_name}'] = []
            performance_history[f'val_{metric_name}'] = []        
            
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file, 
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)
        
        train, val = Utils.remove_paths(self.train_ds), Utils.remove_paths(self.val_ds)
          
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=Utils.METRICS)
        results = self.base_model.fit(train,
                            validation_data=val,
                            epochs=epochs,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_freq=1,
                            class_weight=Utils.get_class_weights(train),
                            verbose=1)
        
        for k in results.history.keys():
            if k.startswith('val_'):
                performance_history[k].extend(results.history[k])
            elif k == 'loss':
                performance_history[k].extend(results.history[k])
            else:
                performance_history[f'train_{k}'].extend(results.history[k])
        self.history = performance_history
        
        os.makedirs('metrics', exist_ok=True)
        with open(f'metrics/Metrics {self.name} for {self.model_id.upper()}{self.version}.txt', 'w') as f:
            json.dump(performance_history, f, indent=4)
    
    def test(self):
        test = Utils.remove_paths(self.test_ds)
        self.actual, self.predicted = Utils.get_actual_predicted_labels(test, self.base_model)
        
        acc = accuracy_score(self.actual, self.predicted)
        balanced_acc = balanced_accuracy_score(self.actual, self.predicted)
        precision = precision_score(self.actual, self.predicted, average='macro', zero_division=0)
        recall = recall_score(self.actual, self.predicted, average='macro')
        f1 = f1_score(self.actual, self.predicted, average='macro')
        report = classification_report(self.actual, self.predicted, target_names=self.label_names, zero_division=0)
        
        with open(f'metrics/Metrics {self.name} for {self.model_id.upper()}{self.version}.txt', 'a') as f:
            f.write(f"\nacc={acc}, balanced_acc={balanced_acc}, precision={precision}, recall={recall}, f1={f1}")
            f.write(f"\n\n{report}")
        
        return balanced_acc
        
    def predict(self, ds):
        labels = self.base_model.predict(ds)
        return labels
        
    def load_best_weights(self):
        self.base_model.load_weights(self.checkpoint_dir+self.weights_file)
    
    def init_base_model(self, causal=False):
        self.base_model = Utils.AIPCreateBackboneAndClassifierModel(model_id=self.model_id, 
                                                         num_classes=self.num_classes, 
                                                         frames_number=self.num_frames, 
                                                         batch_size=self.batch_size, 
                                                         resolution=self.resolution, 
                                                         train_whole_model=self.train_backbone, 
                                                         dropout=self.dropout,
                                                         checkpoint_dir=self.checkpoint_dir,
                                                         regularization=self.regularization,
                                                         causal_conv=causal,
                                                         stream_mode=False)
    
    def init_stream_model(self, causal=True):
        self.stream_model = Utils.AIPCreateBackboneAndClassifierModel(model_id=self.model_id, 
                                                         num_classes=self.num_classes, 
                                                         frames_number=self.num_frames, 
                                                         batch_size=self.batch_size, 
                                                         resolution=self.resolution, 
                                                         train_whole_model=self.train_backbone, 
                                                         dropout=self.dropout,
                                                         checkpoint_dir=self.checkpoint_dir,
                                                         regularization=self.regularization,
                                                         causal_conv=causal,
                                                         stream_mode=True)

        self.stream_model.set_weights(self.base_model.get_weights())
    
    def init_data(self, extension, train_path = "", val_path = "", test_path = ""):
        Utils.create_data_splits(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
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
            
        Utils.LABEL_NAMES = sorted(os.listdir(Utils.TRAIN_FOLDER))
        self.label_names = sorted(os.listdir(Utils.TRAIN_FOLDER))
        self.num_classes = len(Utils.LABEL_NAMES)
            
    # Modified plot_train_val function
    def plot_train_val(self, savefig=True):
        title = f'Training History {self.name} for {self.model_id.upper()}{self.version}'
        try:
            Utils.plot_metrics(self.history, 
                               ['loss', 'accuracy', 'precision', 'recall', 'f1'], 
                               title, savefig)

        except Exception as e:
            print(f'History not available, not making plots... {e}')
        
    def plot_confusion_matrix(self, savefig=True):
        try:
            Utils.cm_heatmap(self.actual, self.predicted, self.label_names, savefig, 
                       (f'Confusion Matrix {self.name} for {self.model_id.upper()}{self.version}'))
        except Exception as e:
            print(f"We found the error: {e}\nNow doing testing again...")
            self.test()
            Utils.cm_heatmap(self.actual, self.predicted, self.label_names, savefig, 
                       (f'Confusion Matrix {self.name} for {self.model_id.upper()}{self.version}'))
