import cv2
import numpy as np
import pandas as pd

class ImageReplacer:
    def __init__(self) -> None:
        pass
    

    def _replace_detections_with_new_image(self, base_image: str, detections: pd.DataFrame, new_image_path: str) -> np.ndarray:
        """Replaces detected cats in an image with the dog image."""
        # Load the original image
        img = cv2.imread(base_image)
        new_image = cv2.imread(new_image_path)

        for _, detection in detections.iterrows():
            if detection['name'] == 'cat':
                x1, y1, x2, y2 = (int(detection['xmin']), int(detection['ymin']),
                                  int(detection['xmax']), int(detection['ymax']))

                # Resize dog image to fit the bounding box
                new_image_resized = cv2.resize(new_image, (x2 - x1, y2 - y1))

                # Replace the cat region with the dog image
                img[y1:y2, x1:x2] = new_image_resized

        return img

    def save_modified_image(self, img: np.ndarray, output_path: str) -> None:
        """Saves the modified image to the specified path."""
        cv2.imwrite(output_path, img)
        print(f"Modified image saved to {output_path}")

    def __call__(self, image_path: str, detections: pd.DataFrame, new_image_path:str) -> np.ndarray:
        """Enables the instance to be called like a function to replace cats with dogs."""
        return self._replace_detections_with_new_image(image_path, detections, new_image_path)

# Example usage
if __name__ == "__main__":
    dog_image_path = '/path/to/your/dog_image.jpg'  # Update this to your dog image path
    replacer = ImageReplacer(dog_image_path)

    image_path = '/path/to/your/image_directory/cat_image.jpg'  # Update this to your image path
    detections = pd.read_csv('/path/to/your/image_directory/detections_cat_image.jpg.csv')  # Load saved detections

    # Call the replacer instance to replace cats with dogs
    modified_image = replacer(image_path, detections)
    replacer.save_modified_image(modified_image, '/path/to/save/modified_image.jpg')
