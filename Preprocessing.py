import cv2
import numpy as np
import random
import os
import pandas as pd
import datetime
from moviepy.editor import VideoFileClip

def draw_bounding_box(path):
    # Initialize the video capture object
    cap = cv2.VideoCapture(path)
    
    # Set the video capture object to a random frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count - 1))
    
    # Read the current frame of the video
    ret, frame = cap.read()
    
    # Create a window to display the video frame
    cv2.namedWindow('Video Frame')
    
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count - 1))
            ret, frame = cap.read()
            bbox = None
    
        # If the user has pressed 'r', clear the bounding box
        if key == ord('r'):
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
    
    coordinates = draw_bounding_box(input_path)


    # Find the number of times to rotate the video by 90 degrees
    rotation_count = 0
    while True:
        # Read the current frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame by the current number of rotations
        rotated_frame = rotate_frame(frame, rotation_count)

        # Display the rotated frame and wait for user input
        cv2.namedWindow("Rotate", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Rotate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Rotate', rotated_frame)
        key = cv2.waitKey(0) & 0xFF

        # If the user pressed 'r', increment the number of rotations
        if key == ord('r'):
            rotation_count = (rotation_count + 1) % 4

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
        # if cap.get(cv2.CAP_PROP_POS_FRAMES) == 300:
        #     break

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

def create_snippets(excel_path, data_folder, output_folder):
    # Read Excel file into a dictionary of dataframes, one for each sheet
    sheets = pd.read_excel(excel_path, sheet_name=None)

    for sheet_name, df in sheets.items():
        # Extract video name from sheet name
        video_name = sheet_name.strip()

        # Find all files that end in video_name.mp4 in the output folder
        video_paths = [os.path.join(output_folder, f) for f in os.listdir(data_folder)
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
                    output_path = os.path.join(folder_path, f'{behaviour}_{datetime.datetime.now().strftime("%H_%M_%S")}.mp4')
                    snippet.write_videofile(output_path)

    
    

    
if __name__ == '__main__':
    path = "data/long/long.mp4"
    out_path = "data/slapi/unlabeled/cropped_417_top.mp4"
    
    #crop_rotate_video(path, out_path)
    create_snippets(out_path, 'annotation.xlsx', 'data/slapi/labeled')