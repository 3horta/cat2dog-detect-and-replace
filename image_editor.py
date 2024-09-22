import cv2
import numpy as np
import pandas as pd
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from PIL import Image
from typing import List


class ImageEditor:
    """
    Class for editing images by detecting and replacing objects using inpainting models and image manipulation.

    Attributes:
        config (Config): Configuration object for the LAMA inpainting model.
        inpainting_model (ModelManager): Model manager for the LAMA inpainting model.
    """

    def __init__(self, inp_model_steps: int = 100, lama_model_type: str = 'full') -> None:
        """
        Initializes the ImageEditor class with model steps and LAMA model type.

        Args:
            inp_model_steps (int, optional): Number of steps for the LAMA inpainting model. Defaults to 100.
            lama_model_type (str, optional): Type of LAMA model to use (e.g., 'full'). Defaults to 'full'.
        """
        self.config = Config(        #Initialize the Lama Config with the required fields
            ldm_steps= inp_model_steps,                   
            hd_strategy="resize",             
            hd_strategy_crop_margin=128,   
            hd_strategy_crop_trigger_size=512,  
            hd_strategy_resize_limit=1024   
        )
        self.inpainting_model = ModelManager('lama', device='cpu', model_type= lama_model_type)

    def _replace_detections_with_new_image(self, base_image: str, detections: pd.DataFrame, new_image_path: str) -> np.ndarray:
        """
        Replaces detected objects (e.g., cats) in an image with a new image.

        Args:
            base_image (str): Path to the base image where objects will be replaced.
            detections (pd.DataFrame): DataFrame containing the detections, including bounding box coordinates and class names.
            new_image_path (str): Path to the image that will replace the detected objects.

        Returns:
            np.array: The modified image with the objects replaced.
        """
        img = cv2.imread(base_image)
        new_image = cv2.imread(new_image_path)
        img_shape = img.shape
        img_mask = self._create_mask(img_shape, detections)

        mask_image = Image.fromarray(img_mask)
        mask_image.save('debug/masked_image.jpg')
        
        inpainted_image = self._inpaint_image(base_image, img_mask)

        for _, row in detections.iterrows():
            if row['name'] == 'cat':
                bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                new_image_resized = self._resize_image(new_image, bbox)
                inpainted_image = self._apply_seamless_clone(inpainted_image, new_image_resized, bbox)
        
        return inpainted_image

    def _resize_image(self, image, bbox):
        """
        Resizes the replacement image to fit within the bounding box.

        Args:
            image (np.ndarray): The image to be resized.
            bbox (List[int]): Bounding box coordinates [xmin, ymin, xmax, ymax] specifying the region to replace.

        Returns:
            np.ndarray: The resized image.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        try:
            return cv2.resize(image, (width, height))
        except:
            return image

    def _apply_seamless_clone(self, input_image: np.array, replacement_image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Seamlessly blends the replacement image into the input image using Poisson blending.

        Args:
            input_image (np.ndarray): The input image where the replacement will occur.
            replacement_image (np.ndarray): The image to be inserted into the bounding box.
            bbox (List[int]): Bounding box coordinates [xmin, ymin, xmax, ymax] of the object to be replaced.

        Returns:
            np.ndarray: The modified image with the replacement blended in.
        """

        # Center point of the object to replace
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Create mask for the replacement image (all white)
        mask = 255 * np.ones(replacement_image.shape, replacement_image.dtype)
        # Seamless clone (Poisson blending)
        blended_image = cv2.seamlessClone(replacement_image, input_image, mask, center, cv2.MIXED_CLONE)
        
        return blended_image

    
    def _inpaint_image( self, image_path: str, mask: np.array) -> np.array:
        """
        Inpaints an image using the LAMA model and returns the result.

        Args:
            image_path (str): Path to the input image to be inpainted.
            mask (np.ndarray): Input mask image where white regions (255) represent areas to inpaint.

        Returns:
            np.ndarray: The inpainted image as a NumPy array.
        """

        
        img = Image.open(image_path)

        mask_image = mask.astype(np.uint8)
        mask_image = Image.fromarray(mask_image)


        if img.mode != 'RGB': # Ensure the image is in RGB format
            img = img.convert('RGB')
        if mask_image.mode != 'L': # Ensure the mask is in grayscale
            mask = mask.convert('L')

        img_np = np.array(img)
        mask_np = np.array(mask_image)

        result = self.inpainting_model(img_np, mask_np, self.config)

        result_uint8 = np.round(result).astype(np.uint8)
        #result_corrected_img = Image.fromarray(result_uint8)
       
        return result_uint8

    
    def _create_mask(self, image_shape: tuple[int, int], detections: pd.DataFrame) -> dict[str, np.array]:
        """
        Creates a mask for the detected objects based on their bounding box coordinates.

        Args:
            image_shape (Tuple[int, int]): Shape of the input image as (height, width, channels).
            detections (pd.DataFrame): DataFrame containing detection information, including bounding boxes.

        Returns:
            np.ndarray: A binary mask where the detected objects' bounding boxes are white (255) and the background is black (0).
        """
        height, width, _ = image_shape
        mask = np.zeros((height, width), dtype=np.uint8) # Create a blank mask (all zeros) of the same size as the image
        
        for _, d in detections.iterrows() :
            if d['name'] == 'cat':
                x1, y1, x2, y2 = (int(d['xmin']), int(d['ymin']),
                                int(d['xmax']), int(d['ymax']))    
            mask[y1:y2, x1:x2] = 255

        return mask
    
    def save_modified_image(self, img: np.ndarray, output_path: str) -> None:
        """
        Saves the modified image to the specified path.

        Args:
            img (np.ndarray): The image to save.
            output_path (str): The path where the image will be saved.
        """
        cv2.imwrite(output_path, img)
        print(f"Modified image saved to {output_path}")

    def __call__(self, image_path: str, detections: pd.DataFrame, new_image_path:str) -> np.ndarray:
        """
        Replaces detected objects (e.g., cats) in the input image with a new image and returns the modified image.

        Args:
            image_path (str): Path to the base image.
            detections (pd.DataFrame): DataFrame containing detection information (e.g., bounding boxes, class names).
            new_image_path (str): Path to the image to use as the replacement for detected objects.

        Returns:
            np.ndarray: The modified image with objects replaced.
        """
        return self._replace_detections_with_new_image(image_path, detections, new_image_path)
