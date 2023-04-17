import tensorflow as tf
import Utils
from pathlib import Path
from Baseline import BaselineModel


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
                                                                              extension=extension,
                                                                              frame_step=self.frame_step),
                                                             output_signature = self.output_signature)
            self.meta_val_ds = meta_val_ds.batch(self.batch_size)
        
    def train(self):
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.experimental.SGD()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir+self.weights_file, 
                                                              monitor='val_loss', save_weights_only=True, save_best_only=True)
        
        self.base_model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
        
        for i in range(self.tasks):
            tf.keras.backend.clear_session()
            
            # Generate a list of 5 random class indices
            selected_class_indices = tf.random.shuffle(tf.range(self.meta_classes))[:self.num_classes]
            
            # Filter the train_ds and test_ds datasets to keep only the selected classes
            filtered_train_ds = self.meta_train_ds.unbatch().filter(lambda x, y, path: tf.reduce_any(tf.equal(tf.cast(y, tf.int32), selected_class_indices)))
            filtered_test_ds = self.meta_val_ds.unbatch().filter(lambda x, y, path: tf.reduce_any(tf.equal(tf.cast(y, tf.int32), selected_class_indices)))

            # Reset the class labels of the filtered_train_ds and filtered_test_ds datasets
            def reset_labels(x, y, path):
                new_y = tf.reduce_min(tf.where(tf.equal(selected_class_indices, tf.cast(y, tf.int32))))
                return (x, new_y, path)
            
            filtered_train_ds = filtered_train_ds.map(reset_labels)
            filtered_test_ds = filtered_test_ds.map(reset_labels)
            
            # Batch again 
            filtered_train_ds = filtered_train_ds.batch(self.batch_size)
            filtered_test_ds = filtered_test_ds.batch(self.batch_size)
            
            filtered_train_ds = Utils.remove_paths(filtered_train_ds)
            filtered_test_ds = Utils.remove_paths(filtered_test_ds)
             
            results = self.base_model.fit(filtered_train_ds,
                            validation_data=filtered_test_ds,
                            epochs=1,
                            validation_freq=1,
                            verbose=1)
            
            print(f"---Task {i+1}/{self.tasks}---")

        train = Utils.remove_paths(self.train_ds)
        val = Utils.remove_paths(self.val_ds)

        results = self.base_model.fit(train,
                        validation_data=val,
                        epochs=self.epochs,
                        callbacks=[model_checkpoint, early_stopping],
                        validation_freq=1,
                        verbose=1)
        
        return results
