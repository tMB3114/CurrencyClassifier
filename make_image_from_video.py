import cv2
import os


video_path = 'c:/Users/Bhabuk/Desktop/currency detector/Nepali currency/VID_20231129_110722.mp4'  #input path
cap = cv2.VideoCapture(video_path) # read video from specified path

#name of output folder
output_folder = 'c:/Users/Bhabuk/Desktop/currency mine one/ten_me' # output path
os.makedirs(output_folder, exist_ok=True)

# Read and save each frame
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break  

    if frame_count % 7 == 0: # save every 7th frace

        frame_name = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame) # to save frame to output path

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames saved to {output_folder}")
