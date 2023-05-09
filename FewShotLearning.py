import tensorflow as tf
import Utils
from pathlib import Path
from Baseline import BaselineModel
import json


class FewShotModel(BaselineModel):
    
    meta_train_ds: tf.data.Dataset
    meta_val_ds: tf.data.Dataset
    
    meta_classes: int
    tasks: int
    
    def __init__(self, 
                 tasks, meta_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "FSL"
        self.weights_file = f'/movinet_{self.name}_{self.model_id}_{self.model_type}_weights.hdf5'
        self.tasks = tasks
        self.meta_classes = meta_classes
        
    def init_meta_data(self, extension, train_path = "", val_path = ""):
        if train_path != "":
            meta_train_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(train_path), self.num_frames, 
                                                                           training = True, 
                                                                           extension=extension,
                                                                           resolution = self.resolution,
                                                                           shots=self.shots,
                                                                           frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.meta_train_ds = meta_train_ds.batch(self.batch_size)
        
        if val_path != "":
            meta_val_ds = tf.data.Dataset.from_generator(Utils.FrameGenerator(Path(val_path), self.num_frames, 
                                                                           resolution = self.resolution,
                                                                           shots=self.shots,
                                                                           extension=extension,
                                                                           frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.meta_val_ds = meta_val_ds.batch(self.batch_size)
        
    def train(self, learning_rate=1e-3, epochs=5):
        ph = {m.name: [] for m in Utils.METRICS}
        performance_history = {'loss': [], 'val_loss': []}
        
        for metric_name in ph.keys():
            performance_history[f'train_{metric_name}'] = []
            performance_history[f'val_{metric_name}'] = []  
            
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file, 
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)
        
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=Utils.METRICS)

        def reset_labels(x, y, path):
            new_y = tf.reduce_min(tf.where(tf.equal(selected_class_indices, tf.cast(y, tf.int32))))
            return (x, new_y, path)
        
        def filter_func(x, y, path):
            return tf.reduce_any(tf.equal(tf.cast(y, tf.int32), selected_class_indices))

        for i in range(self.tasks):
            tf.keras.backend.clear_session()

            selected_class_indices = tf.random.shuffle(tf.range(self.meta_classes))[:self.num_classes]
            
            filtered_train_ds = self.meta_train_ds.unbatch().filter(filter_func).map(reset_labels).batch(self.batch_size)

            _ = self.base_model.fit(Utils.remove_paths(filtered_train_ds),
                                          epochs=1, verbose=1)
            print(f"---Task {i+1}/{self.tasks}---")
            
        train, val = Utils.remove_paths(self.train_ds), Utils.remove_paths(self.val_ds)
            
        self.base_model.optimizer.lr = learning_rate
        results = self.base_model.fit(train,
                                      validation_data=val,
                                      epochs=epochs,
                                      callbacks=[model_checkpoint, early_stopping],
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
        
        with open(f'metrics/Metrics {self.name} for {self.model_id.upper()}.txt', 'w') as f:
            json.dump(performance_history, f, indent=4)
