# Video Classification with Machine Learning Models

This project provides a script for running machine learning models with different approaches for video classification. The script supports both baseline and active learning models. If you struggle with anything, don't hesitate to contact me.

## Requirements

- Python 3.10.11
- OpenCV 4.7.0.72
- TensorFlow 2.11.1
- tf-models-official 2.11.5
- Other dependencies listed in the `requirements.txt` file

## Setup

Clone the repository:

    ```bash
    git clone https://github.com/joris-s/cues
    ```
### 1. Install Anaconda3 and Python

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-0-Linux-x86_64.sh
bash Anaconda3-2023.03-0-Linux-x86_64.sh
```
### During installation:

- Change the installation location to the `slapi` folder, preferably in a subfolder you created there. Not your own working directory, for example: `[PATH_TO_YOUR_FOLDER]/anaconda3`

- When asked, "Do you wish the installer to initialize Anaconda 3 by running conda init?", answer `yes`. This will add `conda` to bash and start up `conda` upon startup of the terminal.

### 2. Create your own tmp directory due to file size

```bash
export TMPDIR=[PATH_TO_YOUR_FOLDER]/tmp
```

Make sure you create the `tmp` directory beforehand.

### 3. Installing CuDNN

```bash
conda install -c anaconda cudatoolkit
conda install -c anaconda cudnn
python -m pip install tensorflow[and-cuda]
```

### 4. Add to ~/.bashrc

```bash
vim ~/.bashrc
```

Add the following lines:

```bash
export TMPDIR=/hpc/umc_neonatology/slapi/[YOUR_FOLDER]/tmp
conda activate base
cd [PATH_TO_YOUR_FOLDER]
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
```

Save and exit `vim` (press `Esc`, then type `:wq` and press `Enter`).

As final step, make sure the gcc and g++ version are the same by running:

```bash
conda install -c conda-forge gcc
conda install -c conda-forge gxx
```

### 5. Restart the Bash Terminal

Close and reopen your terminal for the changes to take effect.

### 6. Install Required Packages

After restarting the terminal, install any additional packages or dependencies required for your project.

For TensorFlow, install it using:

```bash
python -m pip install tensorflow[and-cuda]
```

The following should cover all remaining packages. 

```bash
python -m pip install -r requirements.txt
```

If you get any `ModuleNotFound: No module name [MODULE] errors`, try:

```bash
python -m pip install [MODULE] # or
conda install [MODULE]
```

## Do a test run: 
    
    ```bash
    python Main.py --help
    ```

## Usage

You can run the script using Dummy mode, and Expert mode. In Dummy mode, you can simply follow the process chart to profile a video, or retrain the model. Before going into detail, it is important to note the following:

- A pretrained baseline model is ready and waiting. You do not have to train the model each time you want to process a video
- The folder _./data/full-video/_ should consist of _only_ the video you want to use. Nothing else. 
- You can preprocess the video in _./data/full-video/_ during the process.

In ```Utils.py``` and ```Preprocessing.py``` you may find a wide range of functions which could enhance your experience. Do not be afraid to call them directly. They are not included in this walkthrough.   

### Dummy mode

To enter Dummy mode, simply enter ```python Main.py``` in your CLI. It will then go through the following steps

1. **_Do you want to preprocess the video placed in ./data/full-video?_** Note that preprocessing is recommended, as the training data was as well. Point the head of the infant north, and crop away any irrelevant background information along the edges of the frame. 
2. **_Do you want to use the standard baseline settings?_** Recommended in dummy mode.
3. **_Do you want to use standard active learning settings?_** if you answered no in the previous Q, you will be prompted to use active learning. If you answer No again, the program exits.
4. **_Do you want to train the model again?_** No is recommended to use the saved weights. If you select Yes, you will start training again, overwriting the saved weights. 
5. **_Do you want to test performance on the test set?_** No is recommended, otherwise you will compute test metrics.
6. **_Do you want to predict/profile the full video?_** 🔴IMPORTANT❗🔴 If you select yes, the video that will be processed is **_./data/full-video/[any name]processed.mp4_**. Make sure that you either preprocessed your file in step 1., or you adjust your filename accordingly. The file must end with _*processed.mp4_

### Expert mode

Here you can select you own parameters using the CL arguments. Step 1. and 6. from the Dummy mode will also be available here. The command line arguments are as follows:

* --model, -m: Specify the model type to run. Options are "bl" (baseline) or "al" (active learning).
* --epochs, -ep: Number of epochs for training (after FSL/AL specials). Default is 5.
* --shots, -sh: Number of meta-training data instances for few-shot learning approach. Use -1 to use all samples. Default is -1.
* --loops, -lo: Number of loops for active learning approach. Default is 3.
* --num-samples, -ns: Number of samples for active learning approach. Default is 50.
* --no-training, -nt: Disable training and just load possible weights instead.
* --train-backbone, -tb: Train the MoViNet backbone as well as the classification head.
* --causal-conv, -cc: Enable causal convolutions.
* --batch-size, -bs: Batch size of datasets for training and testing. Default is 4.
* --clip-length, -cl: Target length of clips, used to compute num_frames (target-fps*cl-1) per snippet. Default is 5.
* --drop-out, -do: Floating-point dropout probability. Default is 0.3.
* --regularization, -rg: Type of regularization for the model. Options are "l2", "None". Default is None.
* --learning-rate, -lr: Floating-point learning rate. Default is 1e-3.

## Examples
```bash
python Main.py
python Main.py --help 
python Main.py -m bl -ep 5 -bs 4 -do 0.3 -cl 5 -sh 20
python main.py -m bl -ep 10 -sh 5 -nt
python main.py --model al --epochs 5 --num-samples 50 --causal-conv
python main.py -m bl -ep 8 --train-backbone --regularization l2
python main.py --model al -lo 3 -ns 100 -nt
python main.py -m bl -ep 5 --clip-length 3 -of
python main.py --model al --batch-size 8 -cc --optical-flow
python main.py -m bl -ep 7 -rg l2 -lr 0.001
python main.py --model al -lo 4 -ns 200 -tb
python main.py -m bl -ep 6 -sh 10 -do 0.2
python main.py --model al --no-training -bs 16
```
