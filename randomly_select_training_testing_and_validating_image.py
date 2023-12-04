import os
import random
import cv2

# Set the paths
input_folder = 'c:/Users/Bhabuk/Desktop/currency mine one/train/twenty'
output_folder = 'c:/Users/Bhabuk/Desktop/currency mine one/folder/test/twenty'  




all_files = os.listdir(input_folder)


num_images_to_select = 100 #to select the number of image

# Randomly select images
selected_files = random.sample(all_files, num_images_to_select)



for file_name in selected_files:
    source_path = os.path.join(input_folder, file_name)
    destination_path = os.path.join(output_folder, file_name)
    img = cv2.imread(source_path)
    cv2.imwrite(destination_path, img)
    os.remove(source_path)

print(f"{num_images_to_select} randomly selected images moved to {output_folder}")
