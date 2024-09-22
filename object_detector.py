import torch
import cv2
from PIL import Image
import pandas as pd
from typing import List, Optional, Union
import numpy as np
from pathlib import Path
import os
from datetime import datetime

class ObjectDetector:
    """
    Class for detecting objects in images using a pre-trained YOLOv5 model.

    Attributes:
        model: The pre-trained YOLOv5 model used for object detection.
    """
    def __init__(self, model_name: str = 'yolov5s') -> None:
        """
        Initializes the object detector with a pre-trained YOLOv5 model.

        Args:
            model_name (str, optional): The name of the YOLOv5 model to use (e.g., 'yolov5s', 'yolov5m'). Defaults to 'yolov5s'.
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def _detect_objects(self, image_path:str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> pd.DataFrame:
        """
        Detects objects in a given image and filters detections based on confidence threshold and class.

        Args:
            image_path (str): Path to the image where objects will be detected.
            threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.3.
            class_filter (Optional[Union[List[str], List[int]]], optional): List of class names (e.g., ['cat', 'dog']) or IDs (e.g., [15, 16]) to filter.
                If None, no class filtering is applied. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing detected objects filtered by class and confidence threshold.
        """
        results = self.model(image_path)
        
        detections = results.pandas().xyxy[0]

        detections = detections[detections['confidence'] >= threshold] if (detections['confidence'] >= threshold).any() else pd.DataFrame()
        if class_filter is not None and not detections.empty:
            detections = detections[detections['name'].isin(class_filter)] 
        
        detections_dict ={}
        detections_dict[image_path.split('/')[-1]]= detections

        return detections
    
    def _process_directory(self, directory_path: str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> dict:
        """
        Detects objects in all images within a directory.

        Args:
            directory_path (str): Path to the directory containing images.
            threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.3.
            class_filter (Optional[Union[List[str], List[int]]], optional): List of class names or IDs to filter detections. Defaults to None.

        Returns:
            dict: A dictionary where keys are image filenames and values are DataFrames with detected objects.
        """

        detections_dict = {}

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
                image_path = os.path.join(directory_path, filename)
                detections = self._detect_objects(image_path, threshold, class_filter)
                detections_dict[filename] = detections

        return detections_dict
    
    def draw_detections(self, images_path:str, detections: dict) -> list[cv2.Mat]:
        """
        Draws bounding boxes around detected objects on the images.

        Args:
            images_path (str): Path to the images or directory containing images.
            detections (dict): A dictionary where keys are image filenames and values are DataFrames with detected objects.

        Returns:
            list[cv2.Mat]: A list of images with drawn bounding boxes around detected objects.
        """
        result={}
        for image_name, detection in detections.items():
            if len(images_path.split('/')[-1].split('.'))>1:
                img =  cv2.imread(f"{images_path}")
            else: img = cv2.imread(f"{images_path}/{image_name}")
            for _, d in detection.iterrows():
                x1, y1, x2, y2 = int(d['xmin']), int(d['ymin']), int(d['xmax']), int(d['ymax'])
                label = d['name']
                confidence = d['confidence']
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label text
                text = f"{label} ({confidence:.2f})"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            result[image_name]= img

        return result

    def save_image(self, images: dict, output_path: str) -> None:
        """
        Saves images with drawn bounding boxes to the specified output directory.

        Args:
            images (dict): A dictionary where keys are image filenames and values are images with drawn bounding boxes.
            output_path (str): The path where the images will be saved.
        """
        for image_name, img in images.items():
            image_name = image_name.split('/')[-1]
            cv2.imwrite(f"{output_path}/{image_name}", img)

    def __call__(self, input_path: Union[str, List[str]], threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None, save_detections_path:Optional[str] = "") -> dict:
        """
        Detects objects in the input image(s) and optionally saves the detection results.

        Args:
            input_path (Union[str, List[str]]): Path to the input image or directory containing images.
            threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.3.
            class_filter (Optional[Union[List[str], List[int]]], optional): List of class names or IDs to filter detections. Defaults to None.
            save_detections_path (Optional[str], optional): Path to save detection results as a CSV file. Defaults to "".

        Returns:
            dict: A dictionary containing detected objects, where keys are image filenames and values are DataFrames with detection results.
        """
        detections_dict={}
        if os.path.isdir(input_path):
            detections_dict= self._process_directory(input_path, threshold, class_filter)
        else:
            detections= self._detect_objects(input_path, threshold, class_filter)
            detections_dict[input_path.split('/')[-1]]= detections

        if save_detections_path:
            # Combine all detections into a single DataFrame
            combined_detections = pd.concat([df.assign(image_file=image_file) for image_file, df in detections_dict.items()])
            
            # Save the combined detections to a single CSV file
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_detections.to_csv(f"{save_detections_path}/all_detections_{current_time}.csv", index=False)

        return detections_dict
