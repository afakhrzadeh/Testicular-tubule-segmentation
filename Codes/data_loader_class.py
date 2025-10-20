from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segment_anything.utils.transforms import ResizeLongestSide

class HistopathologyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, dataset_type, object_dir, transform=None, resize=1024):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        with open(object_dir, "rb") as f:
            self.mask_objects_df = pd.read_csv(object_dir)
        self.dataset_type = dataset_type
        self.transform = transform
        self.resize = resize
        self.Resize_sam_object = ResizeLongestSide(self.resize)
        self.full_transform = True
        self.simple_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # Convert to PyTorch tensor
        ],)
        
    def __len__(self):
        # return len(self.image_filenames)
        return len(self.mask_objects_df)

    def __getitem__(self, idx):
        # Extracting the object from dataframe
        row_of_data = self.mask_objects_df.loc[idx]
        file_name = row_of_data.file_name
        # print(f"\n===> {self.images_dir, file_name} \n")
        # Load image and mask
        image_path = os.path.join(self.images_dir, file_name)
        if self.dataset_type == "RAI_dataset":
            mask_path_with_lumen = os.path.join(self.masks_dir, "mask+", file_name)
            mask_path = os.path.join(self.masks_dir, "mask++lumen", file_name)
        elif self.dataset_type== "Glas_dataset":
            mask_path = os.path.join(self.masks_dir, file_name)
            
        # Check if the image file exists
        if not os.path.exists(image_path):
            raise AssertionError(f"Image file does not exist: {image_path}")

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            raise AssertionError(f"Mask file does not exist: {mask_path}")
        
        # Open image and mask directly using Pillow
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Convert mask to a NumPy array with float32 type
        mask = np.array(mask, np.float32)  
        

        # Preprocess mask file
        if self.dataset_type == "RAI_dataset":
            mask_with_lumen = Image.open(mask_path_with_lumen).convert("L")
            # Convert mask to a NumPy array with float32 type
            mask_with_lumen = np.array(mask_with_lumen, np.float32) 
            mask = self.RAI_data_semantic_mask_binary(mask)
            mask_with_lumen = self.RAI_data_semantic_mask_binary(mask_with_lumen)
        elif self.dataset_type== "Glas_dataset":
            mask = self.glas_semantic_mask_binary(mask)

        # Isolate mask object in the mask file
        if row_of_data.label != 80:
            if self.dataset_type == "RAI_dataset":
                isolated_mask = self.extract_mask_by_box(mask_with_lumen, row_of_data.label)
            elif self.dataset_type== "Glas_dataset":
                isolated_mask = self.extract_mask_by_box(mask, row_of_data.label)
            isolated_mask = isolated_mask * mask
        else:
            isolated_mask = mask
        # Add transformation
        if self.transform:

            points = [(row_of_data['f_point_x'] ,row_of_data['f_point_y']), (row_of_data['b_point_x'] ,row_of_data['b_point_y'])]
            valid_points = all(x <= isolated_mask.shape[1] and y <= isolated_mask.shape[0] for x, y in points)
            
            if self.full_transform and valid_points:
                if row_of_data['label'] != 80:
                    bbox = [row_of_data['x_min'], row_of_data['y_min'], row_of_data['x_max'], row_of_data['y_max']]

                    transformed = self.transform(image=np.array(image),
                                                 mask=np.array(isolated_mask, np.float32),
                                                 keypoints=points,
                                                 bboxes=[bbox],category_ids=[0])
                else:
                    bbox = [0, 0, 1, 1]

                    transformed = self.transform(image=np.array(image),
                                                 mask=np.array(isolated_mask, np.float32),
                                                 keypoints=points,
                                                 bboxes=[bbox],category_ids=[0])
                    

                image = transformed["image"]
                isolated_mask = transformed["mask"]
                aug_points = transformed['keypoints']
                aug_bbox = transformed['bboxes'][0] if transformed['bboxes'] else bbox
            
            else:
                transformed = self.simple_transform(image=np.array(image),
                                             mask=np.array(isolated_mask, np.float32))
                image = transformed["image"]
                isolated_mask = transformed["mask"]
        
        if self.full_transform and valid_points:
            box = np.array([aug_bbox])
            # Convert each column to appropriate tensor type
            x_min = torch.tensor(aug_bbox[0], dtype=torch.int)
            y_min = torch.tensor(aug_bbox[1], dtype=torch.int)
            x_max = torch.tensor(aug_bbox[2], dtype=torch.int)
            y_max = torch.tensor(aug_bbox[3], dtype=torch.int)
            h = torch.tensor(row_of_data['h'], dtype=torch.int)
            w = torch.tensor(row_of_data['w'], dtype=torch.int)
            f_point_x = torch.tensor(aug_points[0][0], dtype=torch.int)
            f_point_y = torch.tensor(aug_points[0][1], dtype=torch.int)
            b_point_x = torch.tensor(aug_points[1][0], dtype=torch.int)
            b_point_y = torch.tensor(aug_points[1][1], dtype=torch.int)
            label = torch.tensor(row_of_data['label'], dtype=torch.int)
        else:
            box = np.array([row_of_data.x_min, row_of_data.y_min, row_of_data.x_max, row_of_data.y_max])
            # Convert each column to appropriate tensor type
            x_min = torch.tensor(row_of_data['x_min'], dtype=torch.int)
            y_min = torch.tensor(row_of_data['y_min'], dtype=torch.int)
            x_max = torch.tensor(row_of_data['x_max'], dtype=torch.int)
            y_max = torch.tensor(row_of_data['y_max'], dtype=torch.int)
            h = torch.tensor(row_of_data['h'], dtype=torch.int)
            w = torch.tensor(row_of_data['w'], dtype=torch.int)
            f_point_x = torch.tensor(row_of_data['f_point_x'], dtype=torch.int)
            f_point_y = torch.tensor(row_of_data['f_point_y'], dtype=torch.int)
            b_point_x = torch.tensor(row_of_data['b_point_x'], dtype=torch.int)
            b_point_y = torch.tensor(row_of_data['b_point_y'], dtype=torch.int)
            label = torch.tensor(row_of_data['label'], dtype=torch.int)

        # Combine tensors
        row_tensors = torch.stack([x_min, y_min, x_max, y_max, h, w, f_point_x, f_point_y, b_point_x, b_point_y, label], dim=0)

        return image, isolated_mask, box, row_tensors, row_of_data['file_name']
    
    def RAI_data_semantic_mask_binary(self, segmentation_mask):
        # prepare the RAI data. the 0 and 1 in this data are reversed. So first we should standardize it.
        # background should be -> black
        # objects should be -> white
        binary_mask = np.zeros(segmentation_mask.shape, dtype=int)
        binary_mask[segmentation_mask == 0] = 1
        return binary_mask
     
    def glas_semantic_mask_binary(self, segmentation_mask):
        # prepare the Glas dataset. Here we have multi values for objects greater than 1 to 255. Also 0 for background
        # converting any value greater than 0 to 1
        binary_mask = np.zeros(segmentation_mask.shape, dtype=int)
        binary_mask[segmentation_mask > 0] = 1
        return binary_mask
    
    def extract_mask_by_box(self, mask, label):
        _, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        isolated_mask = np.where(labels == label, 1, 0).astype(np.uint8)

        return isolated_mask


