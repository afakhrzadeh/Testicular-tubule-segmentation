import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from metrics import ConfusionMatrix, dice, jaccard, precision, recall, accuracy, fscore
import random
import numpy as np
from matplotlib.patches import Rectangle

import torch
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import torch.utils as utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from data_loader_class import HistopathologyDataset
from resize_for_encoder import Resize_model, Resize_model_for_mask
from segment_anything import *
# import segment_anything
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from loss_utils import calculate_loss, Accuracy, get_fp_fn

from segment_anything.trainer import SamTrainer

class run_train_cer():
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
             # k-fold valve
            self.fold = self.config_file["TRAIN"]["Fold"]
            # seed value
            self.seed = self.config_file["TRAIN"]["Seed"]
            # Set device for training
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(f"================> self.device {self.device}")
            # batch size value
            self.batch_size = self.config_file["TRAIN"]["BATCH_SIZE"]
            # resize functions
            self.resize_module = Resize_model().to(self.device)
            self.resize_module_for_mask = Resize_model_for_mask().to(self.device)
            # Transformations
            # self.transform = albumentations.Compose(
            #     [
            #         albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #         albumentations.pytorch.ToTensorV2()
            #     ] 
            # )
            self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=(0.001, 0.2), contrast_limit=(0.001, 0.2), p=0.2),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()  # Convert to PyTorch tensor
                    ],

                # Add keypoints and bounding boxes
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
            # Set mask objects dataframe directory
            self.mask_objects_df_dir = self.config_file["DATASET"]["MASK_OBJECTS_DF"]
            self.iter_time = self.config_file["TRAIN"]["iter"]
            self.w_bce = self.config_file["TRAIN"]["w_bce"]
            self.w_focal = self.config_file["TRAIN"]["w_focal"]
            self.w_tversky = self.config_file["TRAIN"]["w_tversky"]
            self.w_dice = self.config_file["TRAIN"]["w_dice"]
            self.w_iou = self.config_file["TRAIN"]["w_iou"]
            self.val_split = self.config_file["TRAIN"]["val_split"]
            print(f"\n ===>loss function: loss = {self.w_bce} * bce_loss + {self.w_focal} * focal_loss + {self.w_tversky} * tversky_loss + {self.w_dice} * dice_loss + {self.w_iou} * iou_loss")
            print(f"\n Number of iteration of mask prompt: {self.iter_time}")

            self.set_seed()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_dice = 0.0
        self.best_val_Accuracy = 0.0
        self.best_train_dice = 0.0
        self.best_train_Accuracy = 0.0
        
               
    def set_seed(self):
        """Sets the seed for reproducibility."""
        # Python random seed
        random.seed(self.seed)
        
        # NumPy random seed
        np.random.seed(self.seed)
        
        # PyTorch random seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU
        
        # PyTorch deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        # Environment-level seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
#     def load_data(self):
#         # Initialize dataset and dataloader
#         self.train_dataset = HistopathologyDataset(
#               images_dir = self.config_file["DATASET"]["TRAIN_PATH"],
#               masks_dir = self.config_file["DATASET"]["TRAIN_MASK_PATH"],
#               dataset_type = self.config_file["DATASET"]["TRAIN_TYPE"],
#               object_dir = self.mask_objects_df_dir,
#               transform = self.transform
#         )

#         # Initialize K-Fold
#         kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.seed)

#         # Store loaders for each fold
#         self.fold_loaders = []

#         for fold_index, (train_indices, val_indices) in enumerate(kf.split(self.train_dataset)):
#             # Create subsets for training and validation
#             train_subset = Subset(self.train_dataset, train_indices)
#             val_subset = Subset(self.train_dataset, val_indices)

#             # Create data loaders
#             train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)  
#             val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
#             # Append loaders for this fold
#             self.fold_loaders.append((train_loader, val_loader))

#             print(f"Fold {fold_index + 1}:")
#             print(f"Number of batches in the training loader: {len(train_loader)}     Number of batches in the validation loader: {len(val_loader)}")
            
    def load_data(self):
        # Initialize dataset
        self.train_dataset = HistopathologyDataset(
            images_dir=self.config_file["DATASET"]["TRAIN_PATH"],
            masks_dir=self.config_file["DATASET"]["TRAIN_MASK_PATH"],
            dataset_type=self.config_file["DATASET"]["TRAIN_TYPE"],
            object_dir=self.mask_objects_df_dir,
            transform=self.transform
        )

        # Store loaders for each fold or split
        self.fold_loaders = []

        if self.val_split == 0:
            # Train on 100% of data
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            self.fold_loaders.append((train_loader, None))
            print(f"Training on 100% of the data. Total batches: {len(train_loader)}")
        else:
            # Calculate split sizes
            total_size = len(self.train_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            # Random split
            train_subset, val_subset = random_split(self.train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            self.fold_loaders.append((train_loader, val_loader))
            print(f"Training on {train_size} samples, Validation on {val_size} samples.")
            print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")



    def load_model(self):
        saved_model = torch.load(self.sam_checkpoint)
        neck = {}

        # Separate values
        for key, value in saved_model.items():
            if 'image_encoder.neck.' in key:
                neck[key.replace("image_encoder.neck.","")] = value
            else:
                continue
        torch.save(neck, os.path.join(self.config_file['MODEL']['save_path'], "train_neck.pth"))
        self.neck_checkpoint = self.config_file["MODEL"]["neck_checkpoint"]
        # self.neck_checkpoint = None
        # creating the model using sam_model_registry
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint,
                                                        uni_checkpoint=self.uni_checkpoint,                                                                                     neck_checkpoint=self.neck_checkpoint,
                                                        mode=self.config_file["MODEL"]["mode"],
                                                        model_encoder=self.config_file["MODEL"]["model_encoder"])
        # # freezing the image encoder parameter
        # for param in self.sam.image_encoder.parameters():
        #     param.requires_grad = False

        self.sam.to(self.device)
        self.set_train_parameter()
    
    def warm_up(self, epoch, warm_up_epochs=30, target_lr=0.08):
        if epoch < warm_up_epochs:
            # Calculate the learning rate as a linear function of epoch
            lr_scale = (epoch + 1) / warm_up_epochs
            return lr_scale * target_lr
        else:
            return target_lr
        
    def set_train_parameter(self):
        # Adding prompt_encoder and mask_decoder from sam to parameters to model learning
        self.params_to_train = list(self.sam.prompt_encoder.parameters()) + list(self.sam.mask_decoder.parameters())
        # optimizer should be AdamW based on sam paper page 17
        self.optimizer = AdamW(self.params_to_train,
                                lr=self.config_file["TRAIN"]["LEARNING_RATE"], betas=(0.9, 0.999))
        # self.optimizer = SGD(self.params_to_train, lr=self.config_file["TRAIN"]["LEARNING_RATE"]) 

        self.num_epochs = self.config_file["TRAIN"]["NUM_EPOCHS"]
        #Loss function: focal loss Coefficients
        self.teta = self.config_file["TRAIN"]["teta"] 
        self.alpha = self.config_file["TRAIN"]["alpha"] 
        self.gamma = self.config_file["TRAIN"]["gamma"] 
        self.warm_up_epochs = self.config_file["TRAIN"]["warm_up"] 
        # Initialize the scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.9,
                                                                    patience=10,
                                                                    # factor=0.1,
                                                                    # patience=5,
                                                                    verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # self.warm_up_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.warm_up(epoch, self.warm_up_epochs))

        self.model_trainer = SamTrainer(self.sam)
    
    def save_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.sam.state_dict(), checkpoint_path)
 
    def train_loop(self):
        self.lr_list = []
        for fold_index, (train_loader, valid_loader) in enumerate(self.fold_loaders):
            print(f"Training on fold {fold_index + 1}")
            
            for epoch in tqdm(range(self.num_epochs)):

                # Training Loop
                self.train_process(train_loader, epoch)
                # self.train_plus_process(train_loader, epoch)
                
                
                if self.val_split != 0:
                # Validation Loop  
                    self.eval_process(valid_loader, epoch)
                # if epoch < self.warm_up_epochs:
                #     self.warm_up_scheduler.step()
                self.lr_list.append(self.optimizer.param_groups[0]["lr"])
                print(f'Epoch {epoch+1}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
            break

        print("Training complete")
        self.plot_lr(self.lr_list)
        self.plot_loss_and_acc(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

    def train_process(self, train_loader, epoch):
        self.sam.train()
        train_loss = 0.0
        train_Accuracy= 0.0
        train_loss_iter = 0

        ii = False
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        for batch_idx, (images, masks, box, row_of_df, file_name) in enumerate(train_loader):

            box = np.array(box)
            images, masks = images.to(self.device).float(), masks.to(self.device)
            x_min, y_min, x_max, y_max, h, w, f_point_x, f_point_y, b_point_x, b_point_y, label = row_of_df[0]

            # Resize images
            
            images_1024, images_224 = self.resize_module(images)
            masks_1024 = self.resize_module_for_mask(masks)
            
            self.model_trainer.set_image(images_1024, images.shape)
            if label != 80:
                cases = {
                    1: {'point_coords':  np.array([[int(f_point_x.item()),
                                                    int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': None},
                    2: {'point_coords': None, 'point_labels': None, 'box': box},
                    3: {'point_coords': np.array([[int(f_point_x.item()),
                                                   int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': box},
                }

                if b_point_x <= images.shape[3] and b_point_y <= images.shape[2]:
                    cases[4] =  {'point_coords':  np.array([[int(b_point_x.item()),
                                                             int(b_point_y.item())]]), 'point_labels': np.array([0]), 'box': box}
            else:
                cases = {
                    1: {'point_coords':  None, 'point_labels': None, 'box': None},
                }
                

            # if batch_idx % 100 == 0 and epoch % 5 == 0:
                # print(f"{b_point_x}   {images.shape[3]}  {b_point_y}  {images.shape[2]}")
                # ii = True
                # masks_case_list = [] 
            # masks_case_list = []   
            list_of_masks = [None] * 4
            # cc = True
            for num_iter in range(self.iter_time):
                
                for case_number, config in cases.items():
                    # if cc == True: 
                    masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_trainer.train_model(
                        point_coords=config['point_coords'],
                        point_labels=config['point_labels'],
                        box=config['box'],
                        # mask_input=None,
                        mask_input=list_of_masks[case_number - 1],
                        # mask_input=masks_1024,
                        multimask_output=True,
                        return_logits=True,
                    )
                    # else:
                    #     masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_trainer.train_model(
                    #         point_coords=np.array(sec_points),
                    #         point_labels=np.array(sec_labels),
                    #         box=config['box'],
                    #         # mask_input=None,
                    #         mask_input=list_of_masks[case_number - 1],
                    #         # mask_input=masks_1024,
                    #         multimask_output=True,
                    #         return_logits=True,
                    #     )
                    train_loss_iter += 1
                    list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0])
                    # list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0]*masks)
                    
                    # if batch_idx % 100 == 0 and epoch % 5 == 0:
                    # masks_case_list.append(binary_mask[0].detach().cpu().numpy())

                    loss = calculate_loss(masks[0],
                                          masks_case[0],
                                          iou_predictions[0],
                                          self.device,
                                          self.w_bce,
                                          self.w_focal,
                                          self.w_tversky,
                                          self.w_dice,
                                          self.w_iou)
                    
                    train_Accuracy += Accuracy(binary_mask[0], masks[0])
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # for name, param in self.sam.named_parameters():
                    #     if param.requires_grad and param.grad is not None:
                    #         print(f"{name}: Grad norm {param.grad.norm()}")
                    #     elif param.requires_grad and param.grad is None:
                    #         print(f"{name}: No gradient computed!")

                    # Track the running loss
                    train_loss += loss.item()

                    train_conf_matrix = ConfusionMatrix(masks_case[0], masks, self.device)
                    tp, fp, tn, fn = train_conf_matrix.get_matrix()
                    # Update running totals
                    total_tp += tp
                    total_fp += fp
                    total_tn += tn
                    total_fn += fn
                    
                    # sec_points, sec_labels = get_fp_fn(binary_mask[0], masks[0])
                    # cc = False
                
            # if ii:
            #     ii = False
            #     self.box_new_plot(masks, masks_case_list, [x_min, y_min, w, h], epoch, batch_idx, f_point_x, f_point_y, b_point_x, b_point_y, len(cases), file_name)
            # self.box_new_plot(masks, masks_case_list, [x_min, y_min, w, h], epoch, batch_idx, f_point_x, f_point_y, b_point_x, b_point_y, len(cases), file_name)
        # Calculate metrics for training
        train_conf_matrix_list = [total_tp, total_fp, total_tn, total_fn]
        train_dice = dice(confusion_matrix=train_conf_matrix_list)
        train_jaccard = jaccard(confusion_matrix=train_conf_matrix_list)
        train_precision = precision(confusion_matrix=train_conf_matrix_list)
        train_recall = recall(confusion_matrix=train_conf_matrix_list)
        train_accuracy = accuracy(confusion_matrix=train_conf_matrix_list)
        train_f1 = fscore(confusion_matrix=train_conf_matrix_list)
        
        # Print training metrics
        print(f"\nEpoch [{epoch + 1}/{self.num_epochs}], Iteration: {train_loss_iter}, Train Loss: {train_loss/train_loss_iter:.4f}, "
              f"Dice: {train_dice:.4f}, Jaccard: {train_jaccard:.4f}, Precision: {train_precision:.4f}, "
              f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_Accuracy/train_loss_iter:.4f}")
        
        if epoch >= self.warm_up_epochs and self.val_split == 0:
            self.scheduler.step(train_loss/train_loss_iter)
            
        if ((train_loss/train_loss_iter) < self.best_train_dice or epoch == 0) and self.val_split == 0:
            self.best_train_dice = (train_loss/train_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'train_best_model.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path} based on train loss')
            
        if ((train_Accuracy/train_loss_iter) > self.best_train_Accuracy) and self.val_split == 0:
            self.best_train_Accuracy = (train_Accuracy/train_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'train_best_model_accuracy.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path} based on train Accuracy')
        
        self.train_losses.append(train_loss/train_loss_iter)
        self.train_accuracies.append(train_Accuracy/train_loss_iter)
        
    def eval_process(self, valid_loader, epoch):
        self.sam.eval()
        
        val_loss = 0.0
        val_loss_iter = 0
        val_Accuracy = 0.0
        # val_outputs = []
        # val_targets = []
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        ii = False
        with torch.no_grad():
            for batch_idx, (images, masks, box, row_of_df, file_name) in enumerate(valid_loader):

                images, masks = images.to(self.device).float(), masks.to(self.device)
                box = np.array(box)
                x_min, y_min, x_max, y_max, h, w, f_point_x, f_point_y, b_point_x, b_point_y, label = row_of_df[0]
                # masks = masks.unsqueeze(1)

                # Resize images
                with torch.no_grad():
                    images_1024, images_224 = self.resize_module(images)
                    masks_1024 = self.resize_module_for_mask(masks)
                
                self.model_trainer.set_image(images_1024, images.shape)

                if label != 80:
                    cases = {
                        1: {'point_coords':  np.array([[int(f_point_x.item()),
                                                        int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': None},
                        2: {'point_coords': None, 'point_labels': None, 'box': box},
                        3: {'point_coords': np.array([[int(f_point_x.item()),
                                                       int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': box},
                    }

                    if b_point_x <= images.shape[3] and b_point_y <= images.shape[2]:
                        cases[4] =  {'point_coords':  np.array([[int(b_point_x.item()),
                                                                 int(b_point_y.item())]]), 'point_labels': np.array([0]), 'box': box}
                else:
                    cases = {
                        1: {'point_coords':  None, 'point_labels': None, 'box': None},
                    }
                # if batch_idx % 100 == 0 and epoch == 19:

                
                list_of_masks = [None] * 4
                for num_iter in range(self.iter_time): 
                    if (batch_idx % 100 == 0 and (epoch % 5 == 0 or epoch == self.num_epochs-1)) or (label == 80 and batch_idx % 2 == 0 and (epoch % 10 == 0 or epoch == self.num_epochs-1)):
                        ii = True
                        masks_case_list = [] 
                        
                    for case_number, config in cases.items():
                        masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_trainer.train_model(
                            point_coords=config['point_coords'],
                            point_labels=config['point_labels'],
                            box=config['box'],
                            mask_input=list_of_masks[case_number - 1],
                            # mask_input=None,
                            # mask_input=masks_1024,
                            multimask_output=True,
                            return_logits=True,
                        )
                        val_loss_iter += 1
                        # list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0])
                        list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0]*masks)
                        # if batch_idx % 100 == 0 and epoch == 19:
                        if (batch_idx % 100 == 0 and (epoch % 5 == 0 or epoch == self.num_epochs-1)) or (label == 80 and batch_idx % 2 == 0 and (epoch % 10 == 0 or epoch == self.num_epochs-1)):
                            masks_case_list.append(binary_mask[0].detach().cpu().numpy())

                        loss = calculate_loss(masks[0],
                                              masks_case[0],
                                              iou_predictions[0],
                                              self.device,
                                              self.w_bce,
                                              self.w_focal,
                                              self.w_tversky,
                                              self.w_dice,
                                              self.w_iou)
                        
                        val_Accuracy += Accuracy(binary_mask[0], masks[0])
                        val_loss += loss.item()

                        val_conf_matrix = ConfusionMatrix(masks_case[0], masks, self.device)
                        tp, fp, tn, fn = val_conf_matrix.get_matrix()
                        # Update running totals
                        total_tp += tp
                        total_fp += fp
                        total_tn += tn
                        total_fn += fn
                    
                    
                if ii: 
                    ii = False
                    # self.box_new_plot(masks, masks_case_list, [x_min, y_min, w, h], epoch, num_iter, batch_idx, f_point_x, f_point_y, b_point_x, b_point_y, len(cases), file_name)
                        
        # Calculate metrics for validation
        val_conf_matrix_list = [total_tp, total_fp, total_tn, total_fn]
        val_dice = dice(confusion_matrix=val_conf_matrix_list)
        val_jaccard = jaccard(confusion_matrix=val_conf_matrix_list)
        val_precision = precision(confusion_matrix=val_conf_matrix_list)
        val_recall = recall(confusion_matrix=val_conf_matrix_list)
        val_accuracy = accuracy(confusion_matrix=val_conf_matrix_list)
        val_f1 = fscore(confusion_matrix=val_conf_matrix_list)
        
        # Print validation metrics
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Iteration: {val_loss_iter}, Val Loss: {val_loss/val_loss_iter:.4f}, "
              f"Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_Accuracy/val_loss_iter:.4f}")
        if epoch >= self.warm_up_epochs:
            self.scheduler.step(val_loss / val_loss_iter)
        # Save the best model based on validation loss
        # if val_dice > self.best_val_dice:
            # self.best_val_dice = val_dice
        if (val_loss/val_loss_iter) < self.best_val_dice or epoch == 0:
            self.best_val_dice = (val_loss/val_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'best_model.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        if (val_Accuracy/val_loss_iter) > self.best_val_Accuracy:
            self.best_val_Accuracy = (val_Accuracy/val_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'best_model_accuracy.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path} based on Accuracy')
            

        self.val_losses.append(val_loss/val_loss_iter)
        self.val_accuracies.append(val_Accuracy/val_loss_iter)
        # self.val_accuracies.append(val_accuracy)

        # # Save a checkpoint every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'model_epoch_{epoch + 1}.pth')
        #     self.save_model(checkpoint_path)
        #     print(f'Saved checkpoint to {checkpoint_path}')

    def box_new_plot(self, mask, output, box, epoch, num_iter, i, f_point_x, f_point_y, b_point_x, b_point_y, cases, file_name):

        if cases == 1:
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].imshow(output[0].squeeze(0), cmap='gray')
            ax[0].set_title("All Glands")
        
            ax[1].imshow(mask[0].detach().cpu().numpy(), cmap='gray')
            ax[1].set_title("GT")
            fig.suptitle(file_name, fontsize=16)
            plt.savefig(os.path.join(self.config_file['MODEL']['save_path'], f'train_images/box_uni_box_img_epoch_{epoch}_{num_iter}_{i}.png'))
            plt.close()
            return True
        elif cases == 4:
            fig, ax = plt.subplots(1,5,figsize=(10,5))
        else:
            fig, ax = plt.subplots(1,4,figsize=(10,5))
        
        ax[0].imshow(output[0].squeeze(0), cmap='gray')
        ax[0].scatter(f_point_x, f_point_y, color='green')
        ax[0].set_title("point")
        
        ax[1].imshow(output[1].squeeze(0), cmap='gray')
        ax[1].set_title("box") 
        rect = Rectangle((box[0], box[1]), box[2], box[3], edgecolor='red', facecolor='none', linewidth=2)
        ax[1].add_patch(rect)
        
        ax[2].imshow(output[2].squeeze(0), cmap='gray')
        ax[2].scatter(f_point_x, f_point_y, color='green')
        ax[2].set_title("box and forg point")
        rect = Rectangle((box[0], box[1]), box[2], box[3], edgecolor='red', facecolor='none', linewidth=2)
        ax[2].add_patch(rect)
        ax[2].scatter(f_point_x, f_point_y, color='green')
        if cases == 4:
            ax[3].imshow(output[3].squeeze(0), cmap='gray')
            ax[3].set_title("box and back point")
            rect = Rectangle((box[0], box[1]), box[2], box[3], edgecolor='red', facecolor='none', linewidth=2)
            ax[3].add_patch(rect)
            ax[3].scatter(b_point_x, b_point_y, color='red')
            plt.tight_layout()
            
            ax[4].imshow(mask[0].detach().cpu().numpy(), cmap='gray')
            ax[4].set_title("GT")
        else:
            ax[3].imshow(mask[0].detach().cpu().numpy(), cmap='gray')
            ax[3].set_title("GT")
            
            
            
        fig.suptitle(file_name, fontsize=16)
        plt.savefig(os.path.join(self.config_file['MODEL']['save_path'], f'train_images/box_uni_box_img_epoch_{epoch}_{num_iter}_{i}.png'))
        plt.close()

    def plot_loss_and_acc(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_file['MODEL']['save_path'], f'acc.png'))
        plt.show()
    
    def plot_lr(self, lr):

        plt.figure(figsize=(12, 6))
        plt.plot(lr, label='Learning rate', color='blue') 
        plt.title('Learning Rate Curve') 
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_file['MODEL']['save_path'], f'lr.png'))
        plt.close()
    
    def train_plus_process(self, train_loader, epoch):
        self.sam.train()
        train_loss = 0.0
        train_Accuracy= 0.0
        train_loss_iter = 0
        accumulation_steps = 10
        step_counter = 0 

        loss = 0.0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        for batch_idx, (images, masks, box, row_of_df, file_name) in enumerate(train_loader):

            box = np.array(box)
            images, masks = images.to(self.device).float(), masks.to(self.device)
            x_min, y_min, x_max, y_max, h, w, f_point_x, f_point_y, b_point_x, b_point_y, label = row_of_df[0]

            # Resize images
            
            images_1024, images_224 = self.resize_module(images)
            masks_1024 = self.resize_module_for_mask(masks)
            
            self.model_trainer.set_image(images_1024, images.shape)
            if label != 80:
                cases = {
                    1: {'point_coords':  np.array([[int(f_point_x.item()),
                                                    int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': None},
                    2: {'point_coords': None, 'point_labels': None, 'box': box},
                    3: {'point_coords': np.array([[int(f_point_x.item()),
                                                   int(f_point_y.item())]]), 'point_labels': np.array([1]), 'box': box},
                }

                if b_point_x <= images.shape[3] and b_point_y <= images.shape[2]:
                    cases[4] =  {'point_coords':  np.array([[int(b_point_x.item()),
                                                             int(b_point_y.item())]]), 'point_labels': np.array([0]), 'box': box}
            else:
                cases = {
                    1: {'point_coords':  None, 'point_labels': None, 'box': None},
                }
                
  
            list_of_masks = [None] * 4
            for num_iter in range(self.iter_time):
                
                for case_number, config in cases.items():
                    # if cc == True: 
                    masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_trainer.train_model(
                        point_coords=config['point_coords'],
                        point_labels=config['point_labels'],
                        box=config['box'],
                        # mask_input=None,
                        mask_input=list_of_masks[case_number - 1],
                        # mask_input=masks_1024,
                        multimask_output=True,
                        return_logits=True,
                    )
             
                    list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0])
                    # list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0]*masks)

                    raw_loss  = calculate_loss(masks[0],
                                          masks_case[0],
                                          iou_predictions[0],
                                          self.device,
                                          self.w_bce,
                                          self.w_focal,
                                          self.w_tversky,
                                          self.w_dice,
                                          self.w_iou)
                    
                    train_Accuracy += Accuracy(binary_mask[0], masks[0])
                    # Backward pass and optimization
 
                    scaled_loss = raw_loss / accumulation_steps
                    scaled_loss.backward()

         
                    train_loss += scaled_loss.item()
                    train_loss_iter += 1
                    step_counter += 1

   

                    # Track the running loss

                    train_conf_matrix = ConfusionMatrix(masks_case[0], masks, self.device)
                    tp, fp, tn, fn = train_conf_matrix.get_matrix()
                    # Update running totals
                    total_tp += tp
                    total_fp += fp
                    total_tn += tn
                    total_fn += fn
                    
            if step_counter == accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_counter = 0

        if step_counter != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Calculate metrics for training
        train_conf_matrix_list = [total_tp, total_fp, total_tn, total_fn]
        train_dice = dice(confusion_matrix=train_conf_matrix_list)
        train_jaccard = jaccard(confusion_matrix=train_conf_matrix_list)
        train_precision = precision(confusion_matrix=train_conf_matrix_list)
        train_recall = recall(confusion_matrix=train_conf_matrix_list)
        train_accuracy = accuracy(confusion_matrix=train_conf_matrix_list)
        train_f1 = fscore(confusion_matrix=train_conf_matrix_list)
        
        # Print training metrics
        print(f"\nEpoch [{epoch + 1}/{self.num_epochs}], Iteration: {train_loss_iter}, Train Loss: {train_loss/train_loss_iter:.4f}, "
              f"Dice: {train_dice:.4f}, Jaccard: {train_jaccard:.4f}, Precision: {train_precision:.4f}, "
              f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_Accuracy/train_loss_iter:.4f}")
        
        if epoch >= self.warm_up_epochs and self.val_split == 0:
            self.scheduler.step(train_loss/train_loss_iter)
            
        if ((train_loss/train_loss_iter) < self.best_train_dice or epoch == 0) and self.val_split == 0:
            self.best_train_dice = (train_loss/train_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'train_best_model.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path} based on train loss')
            
        if ((train_Accuracy/train_loss_iter) > self.best_train_Accuracy) and self.val_split == 0:
            self.best_train_Accuracy = (train_Accuracy/train_loss_iter)
            checkpoint_path = os.path.join(self.config_file['MODEL']['save_path'], f'train_best_model_accuracy.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path} based on train Accuracy')
        
        self.train_losses.append(train_loss/train_loss_iter)
        self.train_accuracies.append(train_Accuracy/train_loss_iter)
        
if __name__ == "__main__":

    model_obj = run_train_cer()
    model_obj.load_data()
    model_obj.load_model()
    # model_obj.set_train_parameter()
    model_obj.train_loop()