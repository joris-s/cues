import cv2
import numpy as np
import random
import os
import pandas as pd
import datetime
from moviepy.editor import VideoFileClip
import Utils
from pathlib import Path
import shutil

# Define the starting position for the first line of text
position = (50, 50)  # (x, y) coordinates of the top-left corner of the first line
line_spacing = 30  # Space between each line of text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (0, 255, 0)  # BGR color tuple (green in this case)
thickness = 1

def draw_bounding_box(path):

    instructions = [
    'click-and-drag bounding box',
    'q - (uit)',
    's - (ave)',
    'n - (ew frame)',
    'c - (lear box)'
    ]
    # Initialize the video capture object
    cap = cv2.VideoCapture(path)
    
    # Set the video capture object to a random frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count - 1))
    
    # Read the current frame of the video
    ret, frame = cap.read()
    
    # Create a window to display the video frame
    cv2.namedWindow("Video Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Define a callback function to handle mouse events
    def draw_box(event, x, y, flags, params):
        nonlocal bbox, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            bbox = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                bbox[2] = x - bbox[0]
                bbox[3] = y - bbox[1]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if bbox[2] < 0:
                bbox[0] += bbox[2]
                bbox[2] = abs(bbox[2])
            if bbox[3] < 0:
                bbox[1] += bbox[3]
                bbox[3] = abs(bbox[3])
            params['bbox'] = bbox
    
    # Register the callback function with the window
    cv2.setMouseCallback('Video Frame', draw_box, {'bbox': None})
    
    # Initialize variables for the bounding box
    bbox = None
    drawing = False
    
    # Loop until the user quits the program
    while True:
        # Display the current frame in the window
        display_frame = frame.copy()
        
        # Overlay the instructions on the frame
        for i, instruction in enumerate(instructions):
            # Calculate the position of each line of text based on the line index
            text_position = (position[0], position[1] + i * line_spacing)
            cv2.putText(display_frame, instruction, text_position, font, font_scale, color, thickness)

        # If the user has drawn a bounding box, display it on the frame
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        cv2.imshow('Video Frame', display_frame)
    
        # Wait for a key press or a mouse event
        key = cv2.waitKey(1) & 0xFF
    
        # If the user pressed 'q', quit the program
        if key == ord('q'):
            break
    
        # If the user pressed 'n', set the video capture object to a new random frame
        elif key == ord('n'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count - int(frame_count/10)))
            ret, frame = cap.read()
            #bbox = None
    
        # If the user has pressed 'c', clear the bounding box
        if key == ord('c'):
            bbox = None
    
        # If the user has pressed 's', save the coordinates of the bounding box
        if key == ord('s'):
            if bbox is not None:
                x, y, w, h = bbox
                print('Bounding box coordinates:', (x, y, x + w, y + h))
                break
    
        # Read the next frame of the video
        ret, frame = cap.read()
    
    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    return (x, y, w, h)

def rotate_frame(frame, n):
    """Rotate a frame clockwise by n*90 degrees."""
    return np.rot90(frame, k=(4-n))

def rotate_coordinates(coordinates, n, rotated_frame):
    """Rotate a bounding box clockwise by n*90 degrees."""
    x, y, w, h = coordinates
    if n == 1:
        x, y = y, x
        w, h = h, w
        x = rotated_frame.shape[1] - x - w
    elif n == 2:
        x = rotated_frame.shape[1] - x - w
        y = rotated_frame.shape[0] - y - h
    elif n == 3:
        x, y = y, x
        w, h = h, w
        y = rotated_frame.shape[0] - y - h
    return x, y, w, h

def crop_rotate_video(input_path, output_path):
    # Open the input video file and get some properties
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    coordinates = draw_bounding_box(input_path)

    instructions = [
    'r - (otate infant up)',
    'q - (uit)',
    's - (ave)',
    'n - (ew frame)',
    'c - (lear box)'
    ]
    
    # Find the number of times to rotate the video by 90 degrees
    rotation_count = 0
    while True:
        # Read the current frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame by the current number of rotations
        rotated_frame = rotate_frame(frame, rotation_count).copy()

        # Overlay the instructions on the frame
        for i, instruction in enumerate(instructions):
            # Calculate the position of each line of text based on the line index
            text_position = (position[0], position[1] + i * line_spacing)
            cv2.putText(rotated_frame, instruction, text_position, font, font_scale, color, thickness)

        # Display the rotated frame and wait for user input
        cv2.namedWindow("Rotate", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Rotate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Rotate', rotated_frame)
        key = cv2.waitKey(0) & 0xFF

        # If the user pressed 'r', increment the number of rotations
        if key == ord('r'):
            rotation_count = (rotation_count + 1) % 4

        elif key == ord('q'):
            break

        elif key == ord('n'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count - int(frame_count/10)))

        # If the user pressed 's', break out of the loop
        elif key == ord('s'):
            # Close any open windows
            cv2.destroyAllWindows()
            break

    print(coordinates)
    coordinates = rotate_coordinates(coordinates, rotation_count, rotated_frame)
    print(coordinates)
    # Create a new video writer object with the same properties as the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (coordinates[2], coordinates[3]))

    # Loop through all the frames in the input video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame by the final number of rotations
        rotated_frame = rotate_frame(frame, rotation_count)

        # Crop the frame to the given bounding box
        x, y, w, h = coordinates
        cropped_frame = rotated_frame[y:y+h, x:x+w]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

    # Release the input and output video objects
    cap.release()
    out.release()

def aggregate_videos_by_class():
    video_codes = os.listdir(Utils.LABELED_FOLDER)

    for video_code in video_codes:
        video_code_path = os.path.join(Utils.LABELED_FOLDER, video_code)
        all_path = os.path.join(video_code_path, 'ALL')
        Path(Utils.LABELED_FOLDER+"/ALL").mkdir(parents=True, exist_ok=True)

        for class_name in os.listdir(video_code_path):
            if class_name == 'ALL':  # Skip the 'ALL' folder
                continue

            class_path = os.path.join(video_code_path, class_name)
            for video_file in os.listdir(class_path):
                src = os.path.join(class_path, video_file)
                dest = os.path.join(all_path, f"{class_name}/{video_file}")
                shutil.copy(src, dest)

def create_snippets(excel_path, data_folder, output_folder):
    # Read Excel file into a dictionary of dataframes, one for each sheet
    sheets = pd.read_excel(excel_path, sheet_name=None)

    for sheet_name, df in sheets.items():
        # Extract video name from sheet name
        video_name = sheet_name.strip()

        # Find all files that end in video_name.mp4 in the output folder
        video_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
                       if f.endswith(f'{video_name}.mp4')]

        if not video_paths:
            print(f"No video found for {video_name}")
            continue

        for i, row in df.iterrows():
            time_val = row['Time']
            duration = row['Duration']
            behaviour = row['Behaviour']

            # Create the output folder if it does not exist
            folder_path = os.path.join(output_folder, video_name, behaviour)
            os.makedirs(folder_path, exist_ok=True)

            # Convert time to datetime object
            datetime_val = datetime.datetime.combine(datetime.date.today(), time_val)
            duration_val = datetime.datetime.combine(datetime.date.today(), duration)
            start = (datetime_val.hour, datetime_val.minute, datetime_val.second)
            end = (start[0] + duration_val.hour, start[1] + duration_val.minute, start[2] + duration_val.second)

            # Extract the video snippet using moviepy
            for video_path in video_paths:
                with VideoFileClip(video_path) as video:
                    snippet = video.subclip(start, end)
                    output_path = f'{output_folder}/{video_name}/{behaviour}/{behaviour}_{datetime.datetime.now().strftime("%H_%M_%S")}.mp4'
                    snippet.write_videofile(output_path)
                    
    aggregate_videos_by_class()

def move_snippets_by_split():

    def clear_folder(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    split_file_path = 'data/slapi/SPLIT'
    train_folder = Utils.TRAIN_FOLDER
    val_folder = Utils.VAL_FOLDER
    test_folder = Utils.TEST_FOLDER
    
    # Check if train, val, and test folders exist, and clear or create them accordingly
    for folder in [train_folder, val_folder, test_folder]:
        if os.path.exists(folder):
            clear_folder(folder)
        else:
            os.makedirs(folder)
    
    if os.path.exists(split_file_path) and os.path.getsize(split_file_path) > 0:
        with open(split_file_path, 'r') as split_file:
            for line in split_file:
                video_path, folder = line.strip().split()
    
                # Get the video filename from the path
                video_filename = os.path.basename(video_path)
    
                # Create the destination folder if it doesn't exist
                dest_folder = os.path.join(folder, video_filename)
                Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
                # Copy the video to the corresponding folder
                shutil.copy(video_path, os.path.join(dest_folder, video_filename))
    else:
        print("SPLIT file does not exist or is empty.")

if __name__ == '__main__':  
    crop_rotate_video("C:\\Users\\joris\\OneDrive - Universiteit Utrecht\\ai\\thesis\\code\\data\\full-video\\long.mp4", 
                      "C:\\Users\\joris\\OneDrive - Universiteit Utrecht\\ai\\thesis\\code\\data\\full-video\\long_processed.mp4")
    #create_snippets('annotation_cues_jstab.xlsx', Utils.UNLABELED_FOLDER, Utils.LABELED_FOLDER)