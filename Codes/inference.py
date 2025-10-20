import os
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from metrics import ConfusionMatrix, dice, jaccard, precision, recall, accuracy, fscore
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

import torch
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import torch.utils as utils
import torch.nn.functional as F


from data_loder_class import HistopathologyDataset
from resize_for_encoder import Resize_model, Resize_model_for_mask
from segment_anything import *
# import segment_anything
import albumentations
import albumentations.pytorch
import logging
from loss_utils import Dice_loss, torch_focal_loss

class run_inference_cer():
    def __init__(self):
        # Load configuration
        self.count_res = 0
        
        with open("./config.yaml", "r") as ymlfile:
            self.config_file = yaml.load(ymlfile, Loader=yaml.Loader)
            # Loading sam and UNI checkpoints
            self.sam_checkpoint = self.config_file["MODEL"]["sam_checkpoint"]
            self.uni_checkpoint = self.config_file["MODEL"]["uni_checkpoint"]
            # specifying vit size
            self.model_type = self.config_file["MODEL"]["model_type"]
            # Set device for training
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # batch size value
            self.batch_size = self.config_file["TEST"]["BATCH_SIZE"]
            # resize functions
            self.resize_module = Resize_model().to(self.device)
            self.resize_module_for_mask = Resize_model_for_mask().to(self.device)
            # Transformations
            self.transform = albumentations.Compose(
                [
                    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    albumentations.pytorch.ToTensorV2()
                ]
            )

        self.teta = self.config_file["TRAIN"]["teta"] 
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_dice = 0.0
            
    def load_data(self):
        # Initialize dataset and dataloader
        self.test_dataset = HistopathologyDataset(
                images_dir = self.config_file["DATASET"]["TEST_PATH"],
                masks_dir = self.config_file["DATASET"]["TEST_MASK_PATH"],
                dataset_type = self.config_file["DATASET"]["TRAIN_TYPE"],
                transform = self.transform
        )
        self.test_loader = DataLoader(self.test_dataset, 
                        batch_size=self.batch_size,
                        shuffle=False
                        )
        print(f"Number of batches in the testing loader: {len(self.test_loader)}")

    def load_model(self):
        saved_model = torch.load(self.config_file["TEST"]["checkpoint"])
        image_encoder_data = {}
        other_data = {}
        neck = {}

        # Separate values
        for key, value in saved_model.items():
            if 'image_encoder.model.' in key:
                image_encoder_data[key.replace("image_encoder.model.","")] = value
            elif 'image_encoder.neck.' in key:
                neck[key.replace("image_encoder.neck.","")] = value
            else:
                other_data[key] = value
        torch.save(image_encoder_data, os.path.join(self.config_file['MODEL']['save_path'], "UNI.pth"))
        torch.save(neck, os.path.join(self.config_file['MODEL']['save_path'], "neck.pth"))
        torch.save(other_data, os.path.join(self.config_file['MODEL']['save_path'], "SAM.pth"))
        self.sam_checkpoint = self.config_file["TEST"]["sam_checkpoint"]
        self.uni_checkpoint = self.config_file["TEST"]["uni_checkpoint"]
        self.neck_checkpoint = self.config_file["TEST"]["neck_checkpoint"]
        # creating the model using sam_model_registry
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint,
                                                        uni_checkpoint=self.uni_checkpoint,
                                                        neck_checkpoint=self.neck_checkpoint,
                                                        mode="inference",
                                                        )
        self.sam.to(self.device)

    def test_loop(self):

        self.sam.eval()

        val_loss = 0.0
        val_outputs = []
        val_targets = []
        j = 0
        self.sam.to(self.device)
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.test_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                #adding channel to tensor from B, H, W  --> to B, C, H, W
                masks = masks.unsqueeze(1)

                # Resize images
                with torch.no_grad():
                    images_1024, images_224 = self.resize_module(images)
                    # masks_1024, masks_224 = resize_module(masks)
                    masks_1024 = self.resize_module_for_mask(masks)

                # Generate image embeddings (frozen encoder)
                with torch.no_grad():
                # features = sam.image_encoder(images_224)
                    features = self.sam.image_encoder(images_1024)

                # Generate prompt embeddings
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=masks_1024
                    )
                
                # Predict masks

                # Pass all required arguments to the mask decoder
                outputs, iou_predictions = self.sam.mask_decoder(
                    image_embeddings=features,        # From the image encoder
                    image_pe=self.sam.prompt_encoder.get_dense_pe(), # Positional encoding
                    sparse_prompt_embeddings=sparse_embeddings,  # Sparse prompts (e.g., points/boxes)
                    dense_prompt_embeddings=dense_embeddings,    # Dense prompts (e.g., masks)
                    multimask_output=False                  # Set to True/False 
                )
                # Compute loss
                loss = (Dice_loss(masks_1024, outputs) + self.teta * (torch_focal_loss(masks_1024, outputs))).mean()
                val_loss += loss.item()
                val_outputs.append(outputs)
                val_targets.append(masks)
                predictions = torch.argmax(outputs, dim=1)
                # for img, msk, pred in zip(images, masks, predictions):
                for i, (msk, pred) in enumerate(zip(masks, predictions)):    
                    self.visualize_results(images.squese().cpu().numpy(), msk.cpu().numpy(), pred.cpu().numpy())
                j += 1
        
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)

        # Calculate metrics for validation
        val_conf_matrix = ConfusionMatrix(val_outputs, val_targets)
        tp, fp, tn, fn = val_conf_matrix.get_matrix()
        val_conf_matrix_list = [tp, fp, tn, fn]
        val_dice = dice(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_jaccard = jaccard(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_precision = precision(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_recall = recall(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_accuracy = accuracy(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_f1 = fscore(output=val_outputs, target=val_targets, confusion_matrix=val_conf_matrix_list)
        
        # Print test metrics
        print(f"Test Loss: {val_loss/len(self.test_loader):.4f}, "
              f"Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

    def visualize_results(self, images, masks, preds):
         
        plt.figure(figsize=(12, 10))

        # Original image
        plt.subplot(2, 2, 1)
        # plt.imshow(images.transpose(1, 2, 0))  # Convert from CHW to HWC
        plt.imshow(images) 
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(self.overlay_mask_on_image(images, preds))
        plt.title('Overlay')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(2, 2, 3)
        plt.imshow(masks, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted mask
        plt.subplot(2, 2, 4)
        plt.imshow(preds, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        


        plt.show()
        
    def save_results(self, images, masks, preds, epoch, folder):
        folder += "/result_output"
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save original image
        image_pil = Image.fromarray((images.transpose(1, 2, 0) * 255).astype('uint8'))  # Convert to HWC and uint8
        image_pil.save(os.path.join(folder, f'{epoch}_image.png'))

        # Save ground truth mask
        mask_pil = Image.fromarray((masks * 255).astype('uint8'))  # Convert mask to uint8
        mask_pil.save(os.path.join(folder, f'{epoch}_mask.png'))

        # Save predicted mask
        pred_pil = Image.fromarray((preds * 255).astype('uint8'))  # Convert prediction to uint8
        pred_pil.save(os.path.join(folder, f'{epoch}_pred.png'))
        
        
    def overlay_mask_on_image(self, image, mask, alpha=0.5):

        # overlayed_image = image.copy()
        # overlayed_image[0] = (1 - alpha) * image[0] + alpha * mask 

        mask[mask == 1] = 255
        t_lower = 30
        t_upper = 100
        # image = np.array(image, dtype='uint8').transpose(1, 2, 0)
        # print(image.shape)
        edges = cv2.Canny(np.array(mask, dtype='uint8'), t_lower, t_upper)
        label = np.zeros_like(image)
        label[edges == 255, :] = [255, 0, 0]
        alpha = 0.6
        beta = 1.0 - alpha
        overlayed_image = np.uint8(alpha * label + beta * image)

        return overlayed_image

    
if __name__ == "__main__":

    model_obj = run_inference_cer()
    model_obj.load_data()
    model_obj.load_model()
    model_obj.test_loop()