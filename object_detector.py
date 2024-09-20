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
    def __init__(self, model_name: str = 'yolov5s') -> None:
        """
        Initialize the object detector with a pre-trained YOLOv5 model.
        
        :param model_name: The name of the YOLO model to use (e.g., 'yolov5s', 'yolov5m').
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def _detect_objects(self, image_path:str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> pd.DataFrame:
        """
        Detect objects in an image and filter by class.
        
        :param image_path: Image data. Must be a string path to an image.
        :param threshold: Confidence threshold for detection filtering.
        :param class_filter: List of class names (e.g., ['cat', 'dog']) or IDs (e.g., [15, 16]) to filter.
        :return: A DataFrame of detected objects filtered by class.
        """
        #img = Image.open(image_path)
        results = self.model(image_path)
        
        # Convert results to DataFrame
        detections = results.pandas().xyxy[0]

        # Filter by confidence threshold
        #detections = [detections[i][detections[i]['confidence'] >= threshold] for i in range(len(detections))]
        detections = detections[detections['confidence'] >= threshold] if (detections['confidence'] >= threshold).any() else pd.DataFrame()

        # Filter by class if specified
        if class_filter is not None and not detections.empty:
            detections = detections[detections['name'].isin(class_filter)] 
        
        return detections
    
    # def _process_directory(self, directory_path: str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> pd.DataFrame:
    #     images_path=[]
    #     for filename in os.listdir(directory_path):
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
    #             image_path = os.path.join(directory_path, filename)
    #             images_path.append(image_path)
    #     detections = self._detect_objects(images_path, threshold, class_filter)

    #     for image_file, detections in detections_dict.items():
    #         detections.to_csv(f"{directory_path}/detections_{image_file}.csv", index=False)

    def _process_directory(self, directory_path: str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> dict:
        """Processes all images in a directory and saves detections."""
        detections_dict = {}

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
                image_path = os.path.join(directory_path, filename)
                detections = self._detect_objects(image_path, threshold, class_filter)
                detections_dict[filename] = detections

        return detections_dict
    
    def draw_detections(self, images_path:str, detections: dict) -> list[cv2.Mat]:
        """
        Draw bounding boxes around detected objects on the images.
        
        :return: A list of images with drawn bounding boxes.
        """
        result={}
        for image_name, detection in detections.items():
            img = cv2.imread(f"{images_path}/{image_name}")
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
        Save the images to a file.
    
        """
        for image_name, img in images.items():
            cv2.imwrite(f"{output_path}/{image_name}", img)

    def __call__(self, input_path: Union[str, List[str]], threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None, save_detections_path:Optional[str] = "") -> dict:
        """
        Call method to detect objects when the instance is called directly.
        
        :param image_path: Path to the input image.
        :param threshold: Confidence threshold for detection filtering.
        :param class_filter: List of class names or IDs to filter (e.g., ['cat', 'dog'] or [15, 16]).
        :return: A DataFrame of detected objects filtered by class.
        """
        detections_dict={}
        if os.path.isdir(input_path):
            detections_dict= self._process_directory(input_path, threshold, class_filter)
        else:
            detections= self._detect_objects(image_path, threshold, class_filter)
            detections_dict[input_path]= detections

        if save_detections_path:
            # Combine all detections into a single DataFrame
            combined_detections = pd.concat([df.assign(image_file=image_file) for image_file, df in detections_dict.items()])
            
            # Save the combined detections to a single CSV file
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_detections.to_csv(f"{save_detections_path}/all_detections_{current_time}.csv", index=False)

        return detections_dict


# Example usage
if __name__ == "__main__":
    # Instantiate the detector
    print("Hello")
    detector = ObjectDetector(model_name='yolov5s')
    image_path= 'test_data'
    # Call the detector directly instead of using detect_objects
    detections = detector(image_path, threshold=0.3, class_filter=['cat', 'dog'], save_detections_path = 'results')
    print("Drawing")
    # Draw detections on the image
    img_with_boxes = detector.draw_detections(image_path, detections)
    print("Saving")
    # Save the image
    detector.save_image(img_with_boxes, 'drawn_images')
