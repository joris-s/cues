from Baseline import BaselineModel
from FewShotLearning import FewShotModel
from ActiveLearning import ActiveLearningModel
import Utils
import argparse

parser = argparse.ArgumentParser(description='Run machine learning models with different approaches')

# Define the flags
parser.add_argument('--few-shot-learning', '-f', nargs='+', metavar='MODEL', help='Run few-shot learning model(s) with optional model version(s)')
parser.add_argument('--active-learning', '-a', nargs='+', metavar='MODEL', help='Run active learning model(s) with optional model version(s)')
parser.add_argument('--baseline', '-b', nargs='+', metavar='MODEL', help='Run baseline model(s) with optional model version(s)')

# Define arguments for epochs
parser.add_argument('--epochs-few-shot', '-ef', type=int, default=5, metavar='N', help='Number of epochs for few-shot learning approach (default: 5)')
parser.add_argument('--epochs-active-learning', '-ea', type=int, default=5, metavar='N', help='Number of epochs for active learning approach (default: 5)')
parser.add_argument('--epochs-baseline', '-eb', type=int, default=5, metavar='N', help='Number of epochs for baseline approach (default: 5)')

# Define arguments for few-shot learning
parser.add_argument('--meta-training-task-numbers', '-mt', type=int, default=5, metavar='N', help='Number of meta-training tasks for few-shot learning approach (default: 5)')
parser.add_argument('--shots', '-sh', type=int, default=3, metavar='N', help='Number of meta-training data instances for few-shot learning approach, use -1 to use all samples (default: 3)')

# Define arguments for active-learning
parser.add_argument('--loops', '-l', type=int, default=3, metavar='N', help='Number of loops for active learning approach (default: 3)')
parser.add_argument('--num-samples', '-ns', type=int, default=3, metavar='N', help='Number of samples for active learning approach (default: 3)')

# Add an option for disabling training and loading possible weights instead
parser.add_argument('--no-training', '-nt', action='store_true', help='Disable training and just load possible weights instead')

# Add an option for data batch size
parser.add_argument('--batch-size', '-bs', type=int, default=16, metavar='N', help='Batch size of datasets for training and testing (default: 16)')
parser.add_argument('--frame-step', '-fs', type=int, default=5, metavar='N', help='Step with which frames are skipped in frame generator (default: 5)')
parser.add_argument('--drop-out', '-do', type=float, default=0.5, metavar='P', help='floating-point dropout probability (default: 0.5)')


# Parse the arguments
args = parser.parse_args()

# Print the arguments
print("Arguments:")
print("----------")
print(f"few-shot-learning models: {args.few_shot_learning}")
print(f"active-learning models: {args.active_learning}")
print(f"baseline models: {args.baseline}")
print()
print(f"epochs for few-shot-learning: {args.epochs_few_shot}")
print(f"epochs for active-learning: {args.epochs_active_learning}")
print(f"epochs for baseline: {args.epochs_baseline}")
print()
print(f"meta-training task numbers: {args.meta_training_task_numbers}")
print(f"shots: {args.shots}")
print()
print(f"loops: {args.loops}")
print(f"num-samples: {args.num_samples}")
print()
print(f"no-training: {args.no_training}")
print(f"batch-size: {args.batch_size}")
print(f"frame-step: {args.frame_step}")
print(f"drop-out: {args.drop_out}")
print("----------")

#python Main.py -b a0 a1 a2 a3 a4 a5 -a a3 --epochs-active-learning 5 --loops 1 --num-samples 3 -f a1 a2 a3 --epochs-few-shot 1 --meta-training-task-numbers 5
if __name__ == '__main__':
    
    # Parse the arguments
    args = parser.parse_args()
    
    b_models=[]
    a_models=[]
    f_models=[]
    
    #this setting works well, seems val_loss can keep up with train_loss
    # args.baseline=['a2']
    # args.shots=-1
    # args.frame_step=Utils.MOVINET_PARAMS['a2'][1]
    # args.epochs_baseline=20
    # args.batch_size=16
    # args.drop_out = 0.5
    
    if args.baseline:
        b_models = [BaselineModel(
                    model_id=b_id, model_type="base", 
                    epochs=args.epochs_baseline, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[b_id][0], 
                    num_frames=Utils.MOVINET_PARAMS[b_id][1]*5, 
                    num_classes=12,
                    batch_size=args.batch_size, 
                    frame_step=args.frame_step,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for b_id in args.baseline]
    
    if args.active_learning:
        a_models = [ActiveLearningModel(
                    num_loops=args.loops, num_samples=args.num_samples,
                    data_path='data/long/',
                    model_id=a_id, model_type="base", 
                    epochs=args.epochs_active_learning, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[a_id][0], 
                    num_frames=Utils.MOVINET_PARAMS[a_id][1]*5, 
                    num_classes=12,
                    batch_size=args.batch_size, 
                    frame_step=args.frame_step,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for a_id in args.active_learning]
    
    if args.few_shot_learning:
        f_models = [FewShotModel(
                    tasks=args.meta_training_task_numbers, meta_classes=Utils.META_CLASSES,
                    model_id=f_id, model_type="base", 
                    epochs=args.epochs_few_shot, shots=args.shots, 
                    dropout=args.drop_out, 
                    resolution=Utils.MOVINET_PARAMS[f_id][0], 
                    num_frames=Utils.MOVINET_PARAMS[f_id][1]*5, 
                    num_classes=12,
                    batch_size=args.batch_size,
                    frame_step=args.frame_step,
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES) 
        for f_id in args.few_shot_learning]
    
    models = b_models + a_models + f_models
    
    for model in models:
        
        print(f'Starting on {model.model_id} model of type {model.name}')
        
        model.init_data('.mp4', "data/self/ercan", "data/self/roos", "data/self/joris")
        if model.name == 'FSL':
            model.init_meta_data('.avi', "data/UCF_meta_learning/train", "data/UCF_meta_learning/test", "data/UCF_meta_learning/test")
        model.init_base_model()
        if not args.no_training:
            model.train()
        try:
            model.load_best_weights()
        except:
            print('Weights not found, training instead.')
            model.train()
            model.load_best_weights()
        model.test()
        model.plot_confusion_matrix()
