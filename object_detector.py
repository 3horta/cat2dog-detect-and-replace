import torch
import cv2
from PIL import Image
import pandas as pd
from typing import List, Optional, Union

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov5s') -> None:
        """
        Initialize the object detector with a pre-trained YOLOv5 model.
        
        :param model_name: The name of the YOLO model to use (e.g., 'yolov5s', 'yolov5m').
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def _detect_objects(self, image_path: str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> pd.DataFrame:
        """
        Detect objects in an image and filter by class.
        
        :param image_path: Path to the input image.
        :param threshold: Confidence threshold for detection filtering.
        :param class_filter: List of class names (e.g., ['cat', 'dog']) or IDs (e.g., [15, 16]) to filter.
        :return: A DataFrame of detected objects filtered by class.
        """
        img = Image.open(image_path)
        results = self.model(img)
        
        # Convert results to DataFrame
        detections = results.pandas().xyxy[0]
        
        # Filter by confidence threshold
        detections = detections[detections['confidence'] >= threshold]
        
        # Filter by class if specified
        if class_filter is not None:
            detections = detections[detections['name'].isin(class_filter)]
        
        return detections

    def draw_detections(self, image_path: str, detections: pd.DataFrame) -> cv2.Mat:
        """
        Draw bounding boxes around detected objects on the image.
        
        :param image_path: Path to the input image.
        :param detections: DataFrame of detected objects with bounding boxes.
        :return: The image with drawn bounding boxes.
        """
        img = cv2.imread(image_path)
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            label = detection['name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label text
            text = f"{label} ({confidence:.2f})"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return img

    def save_image(self, img: cv2.Mat, output_path: str) -> None:
        """
        Save the image to a file.
        
        :param img: The image with drawn detections.
        :param output_path: The path where the image will be saved.
        """
        cv2.imwrite(output_path, img)

    def __call__(self, image_path: str, threshold: float = 0.3, class_filter: Optional[Union[List[str], List[int]]] = None) -> pd.DataFrame:
        """
        Call method to detect objects when the instance is called directly.
        
        :param image_path: Path to the input image.
        :param threshold: Confidence threshold for detection filtering.
        :param class_filter: List of class names or IDs to filter (e.g., ['cat', 'dog'] or [15, 16]).
        :return: A DataFrame of detected objects filtered by class.
        """
        return self._detect_objects(image_path, threshold, class_filter)

# Example usage
if __name__ == "__main__":
    # Instantiate the detector
    print("Hello")
    detector = ObjectDetector(model_name='yolov5s')
    image_path= 'test_data'
    # Call the detector directly instead of using detect_objects
    detections = detector(image_path, threshold=0.4, class_filter=['cat', 'dog'])
    print("Drawing")
    # Draw detections on the image
    img_with_boxes = detector.draw_detections(image_path, detections)
    print("Saving")
    # Save the image
    detector.save_image(img_with_boxes, 'output_image.jpg')
