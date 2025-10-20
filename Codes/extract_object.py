from PIL import Image
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

class extract():
    def __init__(self, masks_dir, masks_with_lumen_dir, dataset_type, save_name):
        
        self.dataset_type = dataset_type
        self.masks_dir = masks_dir
        self.masks_with_lumen_dir = masks_with_lumen_dir
        self.mask_filenames = [f for f in os.listdir(self.masks_dir) if f.endswith('.png')]
        self.mask_with_lumen_filenames = [f for f in os.listdir(self.masks_dir) if f.endswith('.png')]
        self.mask_with_lumen_filenames = [
                f for f in self.mask_with_lumen_filenames if f in self.mask_filenames
            ]
        print(f"number of files: {len(self.mask_with_lumen_filenames)}")
        assert len(self.mask_filenames) == len(self.mask_with_lumen_filenames)
        self.save_name = save_name
        columns = ["file_name","x_min","y_min","x_max","y_max","h","w","f_point_x","f_point_y","b_point_x","b_point_y", "label"]
        self.extract_features = pd.DataFrame(columns=columns)

    def getitem(self):
        if self.dataset_type == "RAI_dataset":
            
            # Load image and mask
            print(f"\n===> started \n")
            for mask_file, mask_with_lumen_file in tqdm(zip(self.mask_filenames, self.mask_with_lumen_filenames)):

                mask_path = os.path.join(self.masks_dir, mask_file)
                mask_with_lumen_path = os.path.join(self.masks_with_lumen_dir, mask_with_lumen_file)

                with open(mask_path, "rb") as f:
                    mask = Image.open(f).convert("L")
                mask = np.array(mask, np.float32)  

                with open(mask_with_lumen_path, "rb") as f:
                    mask_with_lumen = Image.open(f).convert("L")
                mask_with_lumen = np.array(mask_with_lumen, np.float32) 

                mask = self.RAI_data_semantic_mask_binary(mask)
                mask_with_lumen = self.RAI_data_semantic_mask_binary(mask_with_lumen)

                self.extract_left_right_bbox(mask_file,
                                             np.array(mask_with_lumen, np.float32),
                                             np.array(mask, np.float32))

            self.extract_features.to_csv(self.save_name, index=False, sep=',')
            print(f"\n===> ended \n")
        else:
            raise ValueError ("The dataset should be RAI_dataset")
    
    def RAI_data_semantic_mask_binary(self, segmentation_mask):
        # prepare the RAI data. the 0 and 1 in this data are reversed. So first we should standardize it.
        # background should be -> black
        # objects should be -> white
        binary_mask = np.zeros(segmentation_mask.shape, dtype=int)
        binary_mask[segmentation_mask == 0] = 1
        return binary_mask
    
    def extract_left_right_bbox(self, mask_file, mask_with_lumen, mask):
        # print(mask_file)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_with_lumen.astype(np.uint8))
        lumen = mask_with_lumen - mask
        lumen_indices = np.argwhere(lumen != 0)
        mask_indices = np.argwhere(mask != 0)
        # Ensuring that there are valid points in both arrays
        if len(mask_indices) > 0:
            for label in range(1, num_labels):
                rows, cols = np.where(labels == label)
                isolated_mask = np.where(labels == label, 255, 0).astype(np.uint8)

                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                w = x_max - x_min
                h = y_max - y_min
                
                if (h * w) > 100: 
                    if len(lumen_indices) > 0:
                        filtered_lumen_indices = [
                            idx for idx in lumen_indices 
                            if y_min <= idx[0] <= y_max and x_min <= idx[1] <= x_max
                        ]
                        if len(filtered_lumen_indices) == 0:
                            # print("\n no lumen \n")
                            lumen_indices_x, lumen_indices_y = mask.shape
                            # Add 1 to x, y to be bigger than the size of the image
                            # we use this to ignore background point in training 
                            filtered_lumen_indices = [(lumen_indices_x + 1, lumen_indices_y + 1)]
                    else:
                        print("\n no lumen at all \n")
                        lumen_indices_x, lumen_indices_y = mask.shape
                        # Add 1 to x, y to be bigger than the size of the image 
                        # we use this to ignore background point in training 
                        filtered_lumen_indices = [(lumen_indices_x + 1, lumen_indices_y + 1)]
                    
                    # Filter mask indices within the specified area
                    filtered_mask_indices = [idx for idx in mask_indices if isolated_mask[idx[0], idx[1]] > 0]
                    
                    if  len(filtered_mask_indices) > 0:

                        point_from_lumen = self.Select_closest_Centroid(filtered_lumen_indices)
                        point_from_mask = filtered_mask_indices[np.random.choice(len(filtered_mask_indices))]
                        b_point_y, b_point_x = point_from_lumen[0], point_from_lumen[1]
                        f_point_y, f_point_x = point_from_mask[0], point_from_mask[1]

                        self.extract_features.loc[len(self.extract_features)] = [mask_file,
                                                                                 x_min,
                                                                                 y_min,
                                                                                 x_max,
                                                                                 y_max,
                                                                                 h,
                                                                                 w,
                                                                                 f_point_x,
                                                                                 f_point_y,
                                                                                 b_point_x,
                                                                                 b_point_y,
                                                                                 label
                                                                                ]

    def Select_closest_Centroid(self, points):
        centroid = np.mean(points, axis=0).astype(int)

        # Find the closest point to the centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        closest_idx = np.argmin(distances)
        selected_point = points[closest_idx]
        return selected_point
        