import os
import cv2


def denoise_image(input_path, output_path, kernel):
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

            # Apply Gaussian blur to remove noise
            denoised_image = cv2.GaussianBlur(image, kernel, 0)

            # Save the denoised image to the output directory
            output_file_path = os.path.join(output_path, file_name)
            cv2.imwrite(output_file_path, denoised_image)


if __name__ == "__main__":
    input_directory = ""
    output_directory = "a noise removal filter "
    kernel_size = (5, 5)  # You can adjust the kernel size based on your needs

    denoise_image(input_directory, output_directory, kernel_size)
