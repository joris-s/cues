# %%
# Imports
import os
import sys
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Baseline import BaselineModel
from ActiveLearning import ActiveLearningModel
import Utils
import Preprocessing

# %%
# Parser
parser = argparse.ArgumentParser(description='Run machine learning models with different approaches')

# Define the flags
parser.add_argument('--model', '-m', type=str, default=None, metavar='TYPE', help='Run baseline (bl) or active learning (al) model.')

# Define arguments for epochs
parser.add_argument('--epochs', '-ep', type=int, default=5, metavar='N', help='Number of epochs for training (after FSL/AL specials) (default: 5)')
parser.add_argument('--shots', '-sh', type=int, default=-1, metavar='N', help='Number of meta-training data instances for few-shot learning approach, use -1 to use all samples (default: -1)')

# Define arguments for active-learning
parser.add_argument('--loops', '-lo', type=int, default=3, metavar='N', help='Number of loops for active learning approach (default: 3)')
parser.add_argument('--num-samples', '-ns', type=int, default=50, metavar='N', help='Number of samples for active learning approach (default: 3)')

# Add an option for disabling training and loading possible weights instead
parser.add_argument('--no-training', '-nt', action='store_true', help='Disable training and just load possible weights instead')
parser.add_argument('--train-backbone', '-tb', action='store_true', help='Train the MoViNet backbone as well as classification head')
parser.add_argument('--causal-conv', '-cc', action='store_true', help='Enable causal convolutions')

# Add an option for data batch size
parser.add_argument('--batch-size', '-bs', type=int, default=4, metavar='N', help='Batch size of datasets for training and testing (default: 16)')
parser.add_argument('--clip-length', '-cl', type=int, default=5, metavar='N', help='Target length of clips, used to compute num_frames(target-fps*cl) per snippet (default: 3)')
parser.add_argument('--drop-out', '-do', type=float, default=0.3, metavar='P', help='Floating-point dropout probability (default: 0.5)')
parser.add_argument('--regularization', '-rg', type=str, default=None, metavar='TYPE', help='Type of regularization for the model (options: "l2", "None"., default: None)')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, metavar='P', help='Floating-point learning rate (default 1e-3)')
parser.add_argument('--optical-flow', '-of', action='store_true', help='Use optical flow preprocessing')

# Parse the arguments
args = parser.parse_args()

# %%
# Print the arguments
print("Arguments:")
print("----------")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("----------")

def train_model(train=False, test=False, bl=True, al=False):
    if al:

        Utils.UNLABELED_PATH = input("Enter the relative path of the preprocessed unlabeled video for AL: ")
        while not os.path.exists(Utils.UNLABELED_PATH):
            print("Path does not exist. Please enter a valid path.")
            Utils.UNLABELED_PATH = input("Enter the relative path of the preprocessed unlabeled video: ")
        
        model = ActiveLearningModel(
            num_loops=args.loops,
            num_samples=args.num_samples,
            unlabeled_path=Utils.UNLABELED_PATH,
            model_id='a2',
            model_type="base",
            shots=args.shots,
            dropout=args.drop_out,
            resolution=Utils.MOVINET_PARAMS['a2'][0],
            num_frames=int(2 * np.floor(Utils.MOVINET_PARAMS['a2'][1] * args.clip_length / 2)),
            num_classes=len(Utils.LABEL_NAMES),
            batch_size=args.batch_size,
            frame_step=int(Utils.FPS / Utils.MOVINET_PARAMS['a2'][1]),
            train_backbone=args.train_backbone,
            regularization=args.regularization,
            output_signature=Utils.OUTPUT_SIGNATURE,
            label_names=Utils.LABEL_NAMES
        )
    elif bl:
        model = BaselineModel(
            model_id='a2',
            model_type="base",
            shots=args.shots,
            dropout=args.drop_out,
            resolution=Utils.MOVINET_PARAMS['a2'][0],
            num_frames=int(2 * np.floor(Utils.MOVINET_PARAMS['a2'][1] * args.clip_length / 2)),
            num_classes=len(Utils.LABEL_NAMES),
            batch_size=args.batch_size,
            frame_step=int(Utils.FPS / Utils.MOVINET_PARAMS['a2'][1]),
            train_backbone=args.train_backbone,
            regularization=args.regularization,
            output_signature=Utils.OUTPUT_SIGNATURE,
            label_names=Utils.LABEL_NAMES
        )
    else:
        print("No model selected, exiting program")
        exit()

    print(f"Starting on {model.model_id} model of type {model.name}")

    model.init_data('.mp4', Utils.TRAIN_FOLDER, Utils.VAL_FOLDER, Utils.TEST_FOLDER)
    if model.name == 'FSL':
        model.init_meta_data('.avi', Utils.META_TRAIN_FOLDER, Utils.META_VAL_FOLDER)

    model.init_base_model(causal=args.causal_conv)
    
    if train:
        model.train(args.learning_rate, args.epochs)
        model.plot_train_val()
        
    try:
        model.load_best_weights()
    except FileNotFoundError:
        print('Weights not found, training instead.')
        model.train(args.learning_rate, args.epochs)
        model.plot_train_val()
        model.load_best_weights()

    if test:
        model.test()
        model.plot_confusion_matrix()

    return model


def get_user_input(prompt):
    return input(prompt).strip().lower()

if __name__ == '__main__':
    model = None
    train = False
    test = False
    bl = True
    al = False

    if len(sys.argv) > 1:
        train = not args.no_training
        test = True
        bl = (args.model == 'bl')
        al = (args.model == 'al')
        model = train_model(train, test, bl, al)

    # Find the video file
    mp4_file = next((os.path.join('data/full-video', file) for file in os.listdir('data/full-video') if file.endswith('.mp4')), None)

    # Check if the video file was found
    if mp4_file:
        print(f"Found video file: {mp4_file}")
    else:
        print("Video file not found")
        if get_user_input("Do you want to proceed anyways? [Y/N]: ") != 'y':
            print("Exiting the program")
            exit()

    if get_user_input("Do you want to preprocess a video? [Y/N]: ") == 'y':
        processed_mp4_file = mp4_file[:-4] + "_processed" + mp4_file[-4:]
        Preprocessing.crop_rotate_video(mp4_file, processed_mp4_file)
    else:
        print("Skipping video preprocessing")

    if len(sys.argv) == 1:
        # Question 2: Do you want to use the standard baseline settings?
        if get_user_input("Do you want to use the standard baseline settings? [Y/N]: ") == 'y':
            pass

        # Question 3: Do you want to use standard active learning settings?
        elif get_user_input("Do you want to use standard active learning settings? [Y/N]: ") == 'y':
            bl = False
            al = True

        # Question 4: Do you want to use different settings?
        elif get_user_input("Do you want to use different settings? [Y/N]: ") == 'y':
            print("You must use the command line arguments. Read the help below: ")
            os.system('python Main.py -- help')
            exit()

        else:
            print("Proceeding with standard baseline settings")

        # Question 5: Do you want to train the model?
        if get_user_input("Do you want to train the model again? [Y/N]: ") == 'y':
            train = True

        # Question 7: Do you want to test performance on the test set?
        if get_user_input("Do you want to test performance on the test set? [Y/N]: ") == 'y':
            test = True

        model = train_model(train, test, bl, al)

    # Question 8: Do you want to predict/profile the full video?
    if get_user_input("Do you want to predict/profile the full video? [Y/N]: ") == 'y':
        print("Predicting/Profiling the full video")
        processed_mp4_file = next((os.path.join('data/full-video', file) for file in os.listdir('data/full-video') if file.endswith('processed.mp4')), None)
        Utils.predict_video(path=processed_mp4_file, model=model)

    print('Exiting...')