import cv2
import pandas as pd
from typing import List, Union, Optional
import os
from object_detector import ObjectDetector
from image_editor import ImageEditor
from datetime import datetime
from pathlib import Path
import numpy as np


class ObjectReplacer:
    """
    Class for detecting and replacing objects in images using a specified object detection model.

    Attributes:
        image_editor (ImageEditor): An instance of the ImageEditor class for editing images.
        object_detector (ObjectDetector): An instance of the ObjectDetector class for detecting objects.
    """
    def __init__(self, object_detection_model_name: str) -> None:
        """
        Initializes the ObjectReplacer class with an object detection model.

        Args:
            object_detection_model_name (str): The name of the object detection model to be used for detecting objects.
        """
        self.image_editor = ImageEditor()  # For editing images
        self.object_detector = ObjectDetector(object_detection_model_name)  # For detecting objects
        #self.modified_images = []  # To store paths of modified images

    def _replace_object_in_image(self, image_path: str, target_class: Optional[Union[List[str], List[int]]], detection_threshold, new_image_path:str, output_path:str = 'edited_images') -> dict:
        """
        Detects objects in an image or directory of images and replaces detected objects with a new image.

        Args:
            image_path (str): The path to the image or directory of images where objects will be detected.
            target_class (Optional[Union[List[str], List[int]]]): The class(es) of objects to be replaced, 
                specified either by class names (str) or class IDs (int). If None, all detected objects are considered.
            detection_threshold (float): The confidence threshold for object detection. Only objects detected with a 
                confidence score greater than or equal to this value will be replaced.
            new_image_path (str): The path to the image that will replace the detected objects.
            output_path (str, optional): The directory where the modified images will be saved. Defaults to 'edited_images'.

        Returns:
            dict: A dictionary with image filenames as keys and the corresponding modified images as values.
        """

        detections = self.object_detector(image_path, detection_threshold, class_filter= target_class)  # Detect objects in the image
        print(detections.keys())
        edited_images ={}

        if os.path.isdir(image_path):
            for image_name in detections.keys():
                modified_image = self.image_editor(os.path.join(image_path, image_name), detections[image_name], new_image_path)
                edited_images[image_name] = modified_image
        else:
            image_name = image_path.split('/')[-1]
            modified_image = self.image_editor(image_path, detections[image_name], new_image_path)
            edited_images[image_name] = modified_image

        output_path = self._save_images(edited_images, output_path)

        return edited_images, output_path

    def _save_images(self, images: dict, output_path: str) -> None:
        """
        Saves the modified images to the specified output directory.

        Args:
            images (dict): A dictionary containing the modified images with filenames as keys and image data as values.
            output_path (str): The directory where the modified images will be saved. If 'edited_images', 
                a timestamped folder will be created.

        Returns:
            str: The final output directory where the modified images are saved.
        """
        if output_path == 'edited_images':
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, current_time)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        for image_name in images.keys():
            _output_path = os.path.join(output_path,f"modified_{image_name}")
            cv2.imwrite(_output_path, images[image_name]) 
        return output_path

        
    def __call__(self, image_paths: List[str], target_class: Optional[Union[List[str], List[int]]], new_image_path:str, detection_threshold:float = 0.3, save_images_path:str = 'edited_images') -> None:
        """
        Processes a list of images, detects and replaces objects, and saves the modified images.

        Args:
            image_paths (List[str]): A list of image file paths to be processed.
            target_class (Optional[Union[List[str], List[int]]]): The class(es) of objects to be replaced, specified either 
                by class names (str) or class IDs (int). If None, all detected objects are considered.
            new_image_path (str): The path to the image that will replace the detected objects.
            detection_threshold (float, optional): The confidence threshold for object detection. Defaults to 0.3.
            save_images_path (str, optional): The directory where the modified images will be saved. Defaults to 'edited_images'.
        """
        self._replace_object_in_image(image_paths, target_class, detection_threshold, new_image_path, save_images_path)


# Example usage
if __name__ == "__main__":
    
    model_name = 'yolov5s'  # Update as needed
    replacer = ObjectReplacer(model_name)

    replacement_image_path = 'dog_images/pug.png'  # Update to your replacement image path
    image_paths = 'test_data/gato.jpg'
    target_class = ['cat']  # Specify the class to replace

    replacer(image_paths, target_class,replacement_image_path) # Specify the class to replace


