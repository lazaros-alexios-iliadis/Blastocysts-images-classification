import os
import cv2
import numpy as np


def correct_tilt(input_path, output_path):
    # Check if the input directory exists
    if not os.path.exists(input_path):
        print(f"Input directory '{input_path}' does not exist.")
        return

    # Check if the output directory exists, create it if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop through the files in the input directory
    for file_name in os.listdir(input_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(input_path, file_name)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

            # Apply Hough Line Transform to detect dominant lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

            if lines is not None:
                # Calculate the average angle of the lines
                angles = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]
                average_angle = np.mean(angles)

                # Rotate the image to correct tilt
                height, width = image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), np.degrees(average_angle), 1)
                rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

                # Save the corrected image to the output directory
                output_file_path = os.path.join(output_path, file_name)
                cv2.imwrite(output_file_path, rotated_image)


if __name__ == "__main__":
    input_directory = ""
    output_directory = ""

    correct_tilt(input_directory, output_directory)
