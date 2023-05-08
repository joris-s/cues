import os
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from Baseline import BaselineModel
from FewShotLearning import FewShotModel
from ActiveLearning import ActiveLearningModel
import Utils


parser = argparse.ArgumentParser(description='Run machine learning models with different approaches')

# Define the flags
parser.add_argument('--few-shot-learning', '-f', nargs='+', metavar='MODEL', help='Run few-shot learning model(s) with optional model version(s)')
parser.add_argument('--active-learning', '-a', nargs='+', metavar='MODEL', help='Run active learning model(s) with optional model version(s)')
parser.add_argument('--baseline', '-b', nargs='+', metavar='MODEL', help='Run baseline model(s) with optional model version(s)')

# Define arguments for epochs
parser.add_argument('--epochs', '-ep', type=int, default=5, metavar='N', help='Number of epochs for training (after FSL/AL specials) (default: 5)')

# Define arguments for few-shot learning
parser.add_argument('--meta-tasks', '-mt', type=int, default=5, metavar='N', help='Number of meta-training tasks for few-shot learning approach (default: 5)')
parser.add_argument('--shots', '-sh', type=int, default=-1, metavar='N', help='Number of meta-training data instances for few-shot learning approach, use -1 to use all samples (default: -1)')

# Define arguments for active-learning
parser.add_argument('--loops', '-lo', type=int, default=3, metavar='N', help='Number of loops for active learning approach (default: 3)')
parser.add_argument('--num-samples', '-ns', type=int, default=3, metavar='N', help='Number of samples for active learning approach (default: 3)')

# Add an option for disabling training and loading possible weights instead
parser.add_argument('--no-training', '-nt', action='store_true', help='Disable training and just load possible weights instead')
parser.add_argument('--train-backbone', '-tb', action='store_true', help='Train the MoViNet backbone as well as classification head')
parser.add_argument('--causal-conv', '-cc', action='store_true', help='Enable causal convolutions')

# Add an option for data batch size
parser.add_argument('--batch-size', '-bs', type=int, default=16, metavar='N', help='Batch size of datasets for training and testing (default: 16)')
parser.add_argument('--clip-length', '-cl', type=int, default=3, metavar='N', help='Target length of clips, used to compute num_frames(target-fps*cl) per snippet (default: 3)')
parser.add_argument('--drop-out', '-do', type=float, default=0.5, metavar='P', help='Floating-point dropout probability (default: 0.5)')
parser.add_argument('--regularization', '-rg', type=str, default=None, metavar='TYPE', help='Type of regularization for the model (options: "l1", "l2", etc., default: None)')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, metavar='P', help='Floating-point learning rate (default 1e-3)')

# Parse the arguments
args = parser.parse_args()

# Print the arguments
print("Arguments:")
print("----------")
print(f"few-shot-learning models: {args.few_shot_learning}")
print(f"active-learning models: {args.active_learning}")
print(f"baseline models: {args.baseline}")
print()
print(f"meta-training task numbers: {args.meta_tasks}")
print(f"shots:          {args.shots}")
print()
print(f"loops:          {args.loops}")
print(f"num-samples:    {args.num_samples}")
print()
print(f"train-backbone: {args.train_backbone}")
print(f"no-training:    {args.no_training}")
print(f'causal-conv:    {args.causal_conv}')
print()
print(f'epochs:         {args.epochs}')
print(f"batch-size:     {args.batch_size}")
print(f"clip-length:    {args.clip_length}")
print(f"drop-out:       {args.drop_out}")
print(f"regularization: {args.regularization}")
print(f"learning-rate:  {args.learning_rate}")
print("----------")

if __name__ == '__main__':

    b_models=[]
    a_models=[]
    f_models=[]
    
    if args.baseline:
        b_models = [BaselineModel(
                    model_id=b_id, model_type="base", 
                    epochs=args.epochs, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[b_id][0], 
                    num_frames=int(2*np.floor(Utils.MOVINET_PARAMS[b_id][1]*args.clip_length/2)), 
                    num_classes=len(Utils.LABEL_NAMES),
                    batch_size=args.batch_size, 
                    frame_step=int(Utils.FPS/Utils.MOVINET_PARAMS[b_id][1]),
                    train_backbone=args.train_backbone,
                    regularization=args.regularization,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for b_id in args.baseline]
    
    if args.few_shot_learning:
        f_models = [FewShotModel(
                    tasks=args.meta_tasks, meta_classes=Utils.META_CLASSES,
                    model_id=f_id, model_type="base", 
                    epochs=args.epochs, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[f_id][0], 
                    num_frames=int(2*np.floor(Utils.MOVINET_PARAMS[f_id][1]*args.clip_length/2)), 
                    num_classes=len(Utils.LABEL_NAMES),
                    batch_size=args.batch_size,
                    frame_step=int(Utils.FPS/Utils.MOVINET_PARAMS[f_id][1]),
                    train_backbone=args.train_backbone,
                    regularization=args.regularization,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for f_id in args.few_shot_learning]

    if args.active_learning:
        a_models = [ActiveLearningModel(
                    num_loops=args.loops, num_samples=args.num_samples,
                    data_path=Utils.UNLABELED_FOLDER,
                    model_id=a_id, model_type="base", 
                    epochs=args.epochs, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[a_id][0], 
                    num_frames=int(2*np.floor(Utils.MOVINET_PARAMS[a_id][1]*args.clip_length/2)), 
                    num_classes=len(Utils.LABEL_NAMES),
                    batch_size=args.batch_size, 
                    frame_step=int(Utils.FPS/Utils.MOVINET_PARAMS[a_id][1]),
                    train_backbone=args.train_backbone,
                    regularization=args.regularization,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for a_id in args.active_learning]
    

    
    models = b_models + f_models + a_models
    
    for model in models:
        
        print(f'Starting on {model.model_id} model of type {model.name}')
        
        model.init_data('.mp4', Utils.TRAIN_FOLDER, Utils.VAL_FOLDER, Utils.TEST_FOLDER)
        if model.name == 'FSL':
            model.init_meta_data('.avi', Utils.META_TRAIN_FOLDER, Utils.META_VAL_FOLDER)
        
        model.init_base_model(causal=args.causal_conv)
        if not args.no_training:
            model.train(args.learning_rate)
            model.plot_train_val()
        try:
            model.load_best_weights()
        except FileNotFoundError:
            print('Weights not found, training instead.')
            model.train(args.learning_rate)
            model.plot_train_val()
            model.load_best_weights()
        model.test()
        model.plot_confusion_matrix()
