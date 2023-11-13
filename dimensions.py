import os
from PIL import Image


def get_minimum_dimensions(folder_path):
    min_width = float('inf')
    min_height = float('inf')

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    min_width = min(min_width, width)
                    min_height = min(min_height, height)
            except (IOError, OSError):
                # Ignore files that cannot be opened as images
                pass

    if min_width == float('inf') or min_height == float('inf'):
        # No valid images found
        return None
    else:
        return min_width, min_height


# Example usage
folder_path = ""
minimum_dimensions = get_minimum_dimensions(folder_path)

if minimum_dimensions:
    print(
        f"The minimum dimensions of images in the folder are: {minimum_dimensions[0]} x {minimum_dimensions[1]} pixels.")
else:
    print("No images found in the folder.")
