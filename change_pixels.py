import cv2
import numpy as np

def process_image(input_image_path, output_image_path):
    """
    Load an image, print original and modified pixels, and save the modified image.
    
    Args:
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to save the modified image.
    """
    # Load an image
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Error: Unable to load image from {input_image_path}")
        return

    print("Original and Modified Pixels (OpenCV):")
    for i in range(min(image.shape[0], 5)):  # Process only first 5x5 pixels for brevity
        for j in range(min(image.shape[1], 5)):
            original_pixel = image[i, j].tolist()
            # Change pixels (example: grayscale conversion)
            gray = sum(original_pixel) // 3
            modified_pixel = [gray, gray, gray]
            image[i, j] = modified_pixel
            print(f"({i}, {j}): Original: {original_pixel}, Modified: {modified_pixel}")

    # Save the modified image
    cv2.imwrite(output_image_path, image)
    print(f"Modified image saved to {output_image_path}")

# Example usage
process_image("twins_playing_football.png", "modified_image.jpg")

