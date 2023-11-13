import os
import cv2
import numpy as np

# Replace 'path_to_your_folder' with the actual path to your folder containing images
folder_path = ''


# Function to read and resize images from the folder
def read_and_resize_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Resize images to a uniform shape
            images.append(img)
    return images


# Function to calculate mean and standard deviation of the images
def calculate_mean_std(images):
    images = np.array(images, dtype=np.float32)
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std


# Main function
def main():
    images = read_and_resize_images(folder_path)
    if len(images) == 0:
        print("No images found in the folder.")
    else:
        mean, std = calculate_mean_std(images)
        print(f"Mean values: {mean}")
        print(f"Standard deviation values: {std}")


if __name__ == '__main__':
    main()
