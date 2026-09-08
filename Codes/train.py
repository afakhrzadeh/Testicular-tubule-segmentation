import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from new_metrics_kfold import dice_score, iou_score
import random
import numpy as np
from matplotlib.patches import Rectangle
import time 

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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

from data_loader_class import HistopathologyDataset
from resize_for_encoder import Resize_model, Resize_model_for_mask
from segment_anything import *
# import segment_anything
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from loss_utils import calculate_loss, Accuracy, get_fp_fn

from segment_anything.trainer import SamTrainer
from segment_anything.sam_inference import SamInferenceEngine

class run_train_cer():

    def __init__(self):
        # Load configuration

        with open("./config_kfold.yaml", "r") as ymlfile:
            self.config_file = yaml.load(ymlfile, Loader=yaml.Loader)
            # Loading sam and UNI checkpoints
            self.sam_checkpoint = self.config_file["MODEL"]["sam_checkpoint"]
            self.uni_checkpoint = self.config_file["MODEL"]["uni_checkpoint"]
            # Using trainable prompt if it is true 
            self.trainable_prompt_flag = self.config_file["MODEL"]["trainable_prompt"]
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
            self.val_transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ],
                                # Add keypoints and bounding boxes
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
            )
            self.new_neck = self.config_file["MODEL"]["new_neck"] 
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
        
        self.saving_path = ""
        self.create_save_path()
        
        self.best_val_dice = 0.0
        self.best_val_Accuracy = 0.0
        self.best_train_loss = 0.0
        self.best_train_Accuracy = 0.0
        self.current_fold = 0
        self.fold_metrics = {
                                'train_loss': [],
                                'train_dice': [],
                                'train_accuracy': [],
                                'train_dice_all_iter': [],
                                'train_IoU': [],
                                'val_loss': [],
                                'val_dice': [],
                                'val_accuracy': [],
                                'val_dice_all_iter': [],
                                'val_IoU': [],
                            }
                 
    
    def create_save_path(self):

        base_path = self.config_file['MODEL']['save_path']
        train_cfg = self.config_file['TRAIN']

        parts = []

        if train_cfg["kfold"]:
            parts.append(f'{self.config_file["DATASET"]["TRAIN_TYPE"]}_{self.config_file["MODEL"]["model_encoder"]}_{train_cfg["Fold"]}_fold')

        loss_weights = {
            "w_bce": "BCE",
            "w_focal": "focal",
            "w_tversky": "tversky",
            "w_dice": "dice",
            "w_iou": "iou"
        }

        for key, name in loss_weights.items():
            if train_cfg[key] > 0:
                parts.append(f'{train_cfg[key]}_{name}')

        if train_cfg["iter"] > 1:
            parts.append(f'{train_cfg["iter"]}_iter')

        if train_cfg["All_gland"]:
            parts.append("just_all_gland")

        parts.append(f'{train_cfg["NUM_EPOCHS"]}_epoch')

        folder_name = "_".join(parts)

        self.saving_path = os.path.join(base_path, folder_name)

        os.makedirs(self.saving_path, exist_ok=True)
        
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
        
    def load_data(self):
        if self.config_file["TRAIN"]["kfold"]:
            self.load_data_with_k_fold()
        else:
            self.load_data_without_k_fold()

    def load_data_with_k_fold(self):
        # Initialize dataset and dataloader
        train_dataset = HistopathologyDataset(
                  images_dir = self.config_file["DATASET"]["TRAIN_PATH"],
                  masks_dir = self.config_file["DATASET"]["TRAIN_MASK_PATH"],
                  dataset_type = self.config_file["DATASET"]["TRAIN_TYPE"],
                  object_dir = self.mask_objects_df_dir,
                  transform = self.transform
                )
        val_dataset = HistopathologyDataset(
                images_dir=self.config_file["DATASET"]["TRAIN_PATH"],
                masks_dir=self.config_file["DATASET"]["TRAIN_MASK_PATH"],
                dataset_type=self.config_file["DATASET"]["TRAIN_TYPE"],
                object_dir=self.mask_objects_df_dir,
                transform=self.val_transform
                )


        # Initialize K-Fold
        # kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.seed)
        self.splits = np.load(self.config_file["TRAIN"]["Fold_split_path"], allow_pickle=True)

        # Store loaders for each fold
        self.fold_loaders = []

        # for fold_index, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        for fold_index, split_data in enumerate(self.splits):
            train_indices = split_data['train']
            val_indices = split_data['val']
            
            # Create subsets for training and validation
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(val_dataset, val_indices)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)  
            valid_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
            # Append loaders for this fold
            self.fold_loaders.append((train_loader, valid_loader))

            print(f"Fold {fold_index + 1}:")
            print(f"Number of batches in the training loader: {len(train_loader)}     Number of batches in the validation loader: {len(valid_loader)}")
            
    def load_data_without_k_fold(self):
        # Initialize dataset
        self.train_dataset = HistopathologyDataset(
            images_dir = self.config_file["DATASET"]["TRAIN_PATH"],
            masks_dir = self.config_file["DATASET"]["TRAIN_MASK_PATH"],
            dataset_type = self.config_file["DATASET"]["TRAIN_TYPE"],
            object_dir = self.mask_objects_df_dir,
            transform = self.transform
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
            valid_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            self.fold_loaders.append((train_loader, valid_loader))
            print(f"Training on {train_size} samples, Validation on {val_size} samples.")
            print(f"Train batches: {len(train_loader)}, Validation batches: {len(valid_loader)}")

    def load_model(self, checkpoint_path=None):
        
        if checkpoint_path is None:
            # -------- Initial model loading (SAM / UNI) --------
            if self.new_neck:
                print(f"====== self.new_neck {self.new_neck}")
                # If we don't want to use SAM's neck weights, we will use this part.
                # We will load the random weights that were saved one time.
                print(f'\n\n UNI neck weights loaded from: {self.config_file["MODEL"]["UNI_random_neck_checkpoint"]}\n\n')
                self.neck_checkpoint = self.config_file["MODEL"]["UNI_random_neck_checkpoint"]
            else:
                # In this specific way, we want to use the weights provided by SAM's neck.
                if self.config_file["MODEL"]["model_encoder"] == "UNI":
                    saved_model = torch.load(self.sam_checkpoint)
                    neck = {}

                    # Separate values
                    for key, value in saved_model.items():
                        if 'image_encoder.neck.' in key:
                            neck[key.replace("image_encoder.neck.","")] = value
                        else:
                            continue
                    torch.save(neck, os.path.join(self.config_file['MODEL']['save_path'], "train_neck.pth"))
                print(f'\n\n ++UNI neck weights loaded from: {self.config_file["MODEL"]["neck_checkpoint"]}\n\n')
           
            # creating the model using sam_model_registry
            self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint,
                                                            uni_checkpoint=self.uni_checkpoint,
                                                            neck_checkpoint=self.config_file["MODEL"]["neck_checkpoint"],
                                                            mode=self.config_file["MODEL"]["mode"],
                                                            model_encoder=self.config_file["MODEL"]["model_encoder"],
                                                            trainable_prompt=self.trainable_prompt_flag,)

            self.sam.to(self.device)
            self.set_train_parameter()
        else:
            # -------- Load trained checkpoint --------
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path,  
                                    uni_checkpoint=None,    
                                    neck_checkpoint=None,
                                    mode="Inference",  
                                    model_encoder=self.config_file["MODEL"]["model_encoder"],
                                    trainable_prompt=self.trainable_prompt_flag,)
            self.sam.to(self.device)
            self.sam.eval()
            self.model_validator = SamInferenceEngine(self.sam)
        
    def set_train_parameter(self):
        
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
            
        if self.new_neck:
            for param in self.sam.image_encoder.neck.parameters():
                param.requires_grad = True
            
            # Adding prompt_encoder and mask_decoder from sam to parameters to model learning
            self.params_to_train = list(self.sam.prompt_encoder.parameters()) + list(self.sam.mask_decoder.parameters()) + list(self.sam.image_encoder.neck.parameters())
            
        else:
            for param in self.sam.image_encoder.neck.parameters():
                param.requires_grad = True
                # Adding prompt_encoder and mask_decoder from sam to parameters to model learning
            self.params_to_train = list(self.sam.prompt_encoder.parameters()) + list(self.sam.mask_decoder.parameters()) + list(self.sam.image_encoder.neck.parameters())
                 
        # optimizer should be AdamW based on sam paper page 17
        self.optimizer = AdamW(self.params_to_train,
                                lr=self.config_file["TRAIN"]["LEARNING_RATE"], weight_decay=1e-4, betas=(0.9, 0.999))

 
        self.num_epochs = self.config_file["TRAIN"]["NUM_EPOCHS"]
        #Loss function: focal loss Coefficients
        self.teta = self.config_file["TRAIN"]["teta"] 
        self.alpha = self.config_file["TRAIN"]["alpha"] 
        self.gamma = self.config_file["TRAIN"]["gamma"] 
        self.warm_up_epochs = self.config_file["TRAIN"]["warm_up"] 
        # Initialize the scheduler

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=10,
                                                                    threshold=1e-4,
                                                                    min_lr=1e-7,
                                                                    # factor=0.1,
                                                                    # patience=5,
                                                                    # verbose=True
                                                                   )

        self.model_trainer = SamTrainer(self.sam)
    
    def save_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.sam.state_dict(), checkpoint_path)

    def check_wich_weights_will_train(self):
        print("--- Checking trainable parameters ---")

        # 1. List all parameters that are actually scheduled to be updated
        trainable_params = [name for name, p in self.sam.named_parameters() if p.requires_grad]

        print(f"Number of trainable parameters: {len(trainable_params)}")

        # 2. Check if there are any parameters in the image_encoder other than the neck
        encoder_params_in_training = [name for name in trainable_params if "image_encoder" in name and "neck" not in name]

        if len(encoder_params_in_training) == 0:
            print("✅ Verified: No parameters from the image_encoder (except the neck) are being trained.")
        else:
            print("⚠️ Warning: The following parameters from the image_encoder are being trained unintentionally:")
            print(encoder_params_in_training)

        # 3. Check if the neck is actually in the list
        neck_params = [name for name in trainable_params if "neck" in name]
        if len(neck_params) > 0:
            print(f"✅ Verified: Neck layers are being trained (Number of layers: {len(neck_params)})")
        else:
            print("❌ Error: Neck layers were not found!")
        
        total_params = sum(p.numel() for p in self.sam.parameters())
        trainable_params = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)

        print("Total parameters:", total_params)
        print("Trainable parameters:", trainable_params)


    def train_loop(self):
        self.lr_list = []
        # Starting code run time
        start_time = time.perf_counter()
        
        for fold_index, (train_loader, valid_loader) in enumerate(self.fold_loaders):
            print(f"Training on fold {fold_index + 1}")

            self.current_fold = fold_index

            self.fold_metrics['train_loss'].append([])
            self.fold_metrics['train_dice'].append([])
            self.fold_metrics['train_dice_all_iter'].append([])
            self.fold_metrics['train_accuracy'].append([])
            self.fold_metrics['train_IoU'].append([])

            self.fold_metrics['val_loss'].append([])
            self.fold_metrics['val_dice'].append([])
            self.fold_metrics['val_dice_all_iter'].append([])
            self.fold_metrics['val_accuracy'].append([])
            self.fold_metrics['val_IoU'].append([])
            
            self.best_val_dice = 0.0
            self.best_val_Accuracy = 0.0
            self.best_train_loss = 0.0
            self.best_train_Accuracy = 0.0

            torch.cuda.empty_cache()
            self.load_model()
            self.check_wich_weights_will_train()
            
            for epoch in tqdm(range(self.num_epochs)):

                # Training Loop
                self.train_process(train_loader, epoch)
                
                self.lr_list.append(self.optimizer.param_groups[0]["lr"])
                
            if self.config_file["TRAIN"]["kfold"] or self.val_split != 0:
            # Validation Loop  
                torch.cuda.empty_cache()
                print(f"\n Starting validation process.....")
                # Loading BEST MODEL
                checkpoint_path = os.path.join(
                    self.saving_path,
                    f'fold_{self.current_fold}_best_model_train_loss.pth'
                )
                print(f"Loading best model from {checkpoint_path}")
                
                self.load_model(checkpoint_path)
                
                print("Running validation once for this fold...")
                self.eval_process(valid_loader, epoch)
                

        print("Training complete")
        # Calculating run time
        end_time = time.perf_counter()
        runtime = end_time - start_time
        seconds = runtime
        minutes = runtime / 60
        hours = int(runtime / 3600)
        hours += 0.01 * (minutes - (hours * 60))  
        # Save run time
        with open(os.path.join(self.saving_path, f'runtime.txt'), "w") as f:
            f.write(f"Seconds: {seconds:.2f}\n")
            f.write(f"Minutes: {minutes:.2f}\n")
            f.write(f"Hours: {hours:.2f}\n")

        self.save_all_kfold_plots()

    def train_process(self, train_loader, epoch):
        self.sam.train()
        train_loss = 0.0
        train_Accuracy= 0.0
        train_loss_iter = 0


        dice_running, all_iter_iou_running, all_iter_dice_running = 0, 0, 0
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
                    masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_trainer.train_model(
                        point_coords=config['point_coords'],
                        point_labels=config['point_labels'],
                        box=config['box'],
                        mask_input=list_of_masks[case_number - 1],
                        multimask_output=True,
                        return_logits=True,
                    )
                    train_loss_iter += 1
                    list_of_masks[case_number - 1] = self.resize_module_for_mask(binary_mask[0])

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
                    # Track the running loss
                    train_loss += loss.item()
                    self.model_trainer.create_features()
                    

                    all_iter_dice_running += dice_score(masks_case[0], masks)
                    all_iter_iou_running  += iou_score(masks_case[0], masks)
            
            dice_running += dice_score(masks_case[0], masks)
                
        # Calculate metrics for training
        train_dice_all_iter = all_iter_dice_running / train_loss_iter
        train_dice = dice_running / len(train_loader)
        train_IoU  = all_iter_iou_running / train_loss_iter
        
        # Print training metrics
        print(f"\nEpoch [{epoch + 1}/{self.num_epochs}], Iteration: {train_loss_iter}, Train Loss: {train_loss/train_loss_iter:.4f}, "
              f"Dice: {train_dice_all_iter:.4f}, IoU: {train_IoU:.4f}, Accuracy: {train_Accuracy/train_loss_iter:.4f}")
        
        if epoch == 0 or (train_loss/train_loss_iter) < self.best_train_loss:
            self.best_train_loss = (train_loss/train_loss_iter)
            checkpoint_path = os.path.join(
                self.saving_path,
                f'fold_{self.current_fold}_best_model_train_loss.pth'
            )
            self.save_model(checkpoint_path)
            print(f'Saved best model based on Train Loss: {(train_loss/train_loss_iter):.4f}')
        
        self.fold_metrics['train_loss'][self.current_fold].append(train_loss / train_loss_iter)
        self.fold_metrics['train_accuracy'][self.current_fold].append(train_Accuracy / train_loss_iter)
        self.fold_metrics['train_dice'][self.current_fold].append(train_dice)
        self.fold_metrics['train_dice_all_iter'][self.current_fold].append(train_dice_all_iter)
        self.fold_metrics['train_IoU'][self.current_fold].append(train_IoU)
        self.scheduler.step()
        self.scheduler.step(train_loss / train_loss_iter)
        
    def eval_process(self, valid_loader, epoch):
        
        val_loss = 0.0
        val_Accuracy = 0.0
        dice_running, all_iter_iou_running, all_iter_dice_running = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, masks, box, row_of_df, file_name) in enumerate(valid_loader):

                images, masks = images.to(self.device).float(), masks.to(self.device)
                box = np.array(box)
                x_min, y_min, x_max, y_max, h, w, f_point_x, f_point_y, b_point_x, b_point_y, label = row_of_df[0]

                # Resize images
                with torch.no_grad():
                    images_1024, images_224 = self.resize_module(images)
                    masks_1024 = self.resize_module_for_mask(masks)
                
                self.model_validator.set_image(images_1024, images.shape)

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

                for case_number, config in cases.items():
                    masks_case, iou_predictions, low_res_masks_case, binary_mask = self.model_validator.predict(
                        point_coords=config['point_coords'],
                        point_labels=config['point_labels'],
                        box=config['box'],
                        mask_input=None,
                        # mask_input=masks_1024,
                        multimask_output=True,
                        return_logits=True,
                    )

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

                    all_iter_dice_running += dice_score(masks_case[0], masks)
                    all_iter_iou_running  += iou_score(masks_case[0], masks)
                    dice_running += dice_score(masks_case[0], masks)   
                        
        # Calculate metrics for validation
        val_dice = dice_running / len(valid_loader)
        val_dice_all_iter = all_iter_dice_running / len(valid_loader)
        val_IoU = all_iter_iou_running / len(valid_loader)
        # Print validation metrics
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss/len(valid_loader):.4f}, "
              f"Dice: {val_dice:.4f}, IoU: {val_IoU:.4f}, Accuracy: {val_Accuracy/len(valid_loader):.4f}")

            
        self.fold_metrics['val_loss'][self.current_fold].append(val_loss / len(valid_loader))
        self.fold_metrics['val_dice'][self.current_fold].append(val_dice)
        self.fold_metrics['val_dice_all_iter'][self.current_fold].append(val_dice_all_iter)
        self.fold_metrics['val_accuracy'][self.current_fold].append(val_Accuracy / len(valid_loader))
        self.fold_metrics['val_IoU'][self.current_fold].append(val_IoU)

    def save_all_kfold_plots(self):

        save_dir = self.saving_path
        num_folds = len(self.fold_metrics['val_dice'])

        # -------------------------------
        # 1) Learning Curves (Train Only)
        # -------------------------------
        for fold in range(num_folds):
            epochs = range(1, len(self.fold_metrics['train_dice'][fold]) + 1)

            fig, ax1 = plt.subplots(figsize=(14, 6))

            # Score metrics (left y-axis)
            ax1.plot(epochs, self.fold_metrics['train_dice'][fold],
                    label='Train Dice', color='blue')
            ax1.plot(epochs, self.fold_metrics['train_accuracy'][fold],
                    label='Train Accuracy', color='green')
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Epoch')
            ax1.set_title(f'Fold {fold} - Train Metrics')
            ax1.grid(True)

            # Loss (right y-axis)
            ax2 = ax1.twinx()
            ax2.plot(epochs, self.fold_metrics['train_loss'][fold],
                    label='Train Loss', color='red', linestyle='--')
            ax2.set_ylabel('Loss')

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir,
                                    f"fold_{fold}_learning_curve_train.png"),
                        dpi=300)
            plt.close()


        # -------------------------------
        # 2) Mean ± Std (Train vs Val)
        # -------------------------------
        train_dice = np.array(self.fold_metrics['train_dice'])
        val_dice   = np.array(self.fold_metrics['val_dice'])
        train_loss = np.array(self.fold_metrics['train_loss'])
        val_loss   = np.array(self.fold_metrics['val_loss'])
        train_IoU = np.array(self.fold_metrics['train_IoU'])
        val_IoU   = np.array(self.fold_metrics['val_IoU'])

        mean_train_dice = np.mean(train_dice, axis=0)
        std_train_dice  = np.std(train_dice, axis=0)
        mean_val_dice   = np.mean(val_dice, axis=0)
        std_val_dice    = np.std(val_dice, axis=0)
        
        mean_train_IoU = np.mean(train_IoU, axis=0)
        std_train_IoU  = np.std(train_IoU, axis=0)
        mean_val_IoU   = np.mean(val_IoU, axis=0)
        std_val_IoU    = np.std(val_IoU, axis=0)
        
        mean_train_loss = np.mean(train_loss, axis=0)
        std_train_loss  = np.std(train_loss, axis=0)
        mean_val_loss   = np.mean(val_loss, axis=0)
        std_val_loss    = np.std(val_loss, axis=0)

        epochs = range(1, len(mean_train_dice) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)

        # --- Dice train ---
        axes[0].plot(epochs, mean_train_dice, color='orange', label='Train Mean Dice')
        axes[0].fill_between(epochs, mean_train_dice - std_train_dice, mean_train_dice + std_train_dice,
                            color='orange', alpha=0.3)
        axes[0].set_title('K-Fold Mean Dice (Train)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Dice')
        axes[0].legend(loc='lower right')
        axes[0].grid(True)

        # --- IoU train ---
        axes[1].plot(epochs, mean_train_IoU, color='steelblue', label='Train Mean IoU')
        axes[1].fill_between(epochs, mean_train_IoU - std_train_IoU, mean_train_IoU + std_train_IoU,
                            color='steelblue', alpha=0.3)
        axes[1].set_title('K-Fold Mean IoU (Train)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend(loc='lower right')
        axes[1].grid(True)

        # --- Loss train ---
        axes[2].plot(epochs, mean_train_loss, color='crimson', label='Train Mean Loss')
        axes[2].fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                            color='crimson', alpha=0.25)
        axes[2].set_title('K-Fold Mean Loss (Train)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend(loc='upper right')
        axes[2].grid(True)


        fig.suptitle('K-Fold Training Curves: Mean ± Std across Folds', fontsize=13)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(os.path.join(save_dir, "mean_std_train_dice_IoU_loss.png"), dpi=300)
        plt.close()


        # -------------------------------
        # 3) Overlay All Folds (Train)
        # -------------------------------
        # --- Train Dice ---
        fig_dice, ax_dice = plt.subplots(figsize=(10, 6))
        for fold in range(num_folds):
            ax_dice.plot(self.fold_metrics['train_dice'][fold], label=f'Fold {fold}')
        ax_dice.set_title('Overlay All Folds: Train Dice')
        ax_dice.set_xlabel('Epoch')
        ax_dice.set_ylabel('Dice')
        ax_dice.legend()
        ax_dice.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "overlay_train_dice.png"), dpi=300)
        plt.close()

        # --- Train IoU ---
        fig_iou, ax_iou = plt.subplots(figsize=(10, 6))
        for fold in range(num_folds):
            ax_iou.plot(self.fold_metrics['train_IoU'][fold], label=f'Fold {fold}')
        ax_iou.set_title('Overlay All Folds: Train IoU')
        ax_iou.set_xlabel('Epoch')
        ax_iou.set_ylabel('IoU')
        ax_iou.legend()
        ax_iou.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "overlay_train_IoU.png"), dpi=300)
        plt.close()

        # --- Train Loss ---
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        for fold in range(num_folds):
            ax_loss.plot(self.fold_metrics['train_loss'][fold], label=f'Fold {fold}')
        ax_loss.set_title('Overlay All Folds: Train Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "overlay_train_loss.png"), dpi=300)
        plt.close()


        # -------------------------------
        # 4) Best Dice per Fold (Train vs Val)
        # -------------------------------
        best_train_dice_per_fold = [max(self.fold_metrics['train_dice'][fold]) for fold in range(num_folds)]
        best_val_dice_per_fold = [
            max(self.fold_metrics['val_dice'][fold]) for fold in range(num_folds)
        ]

        best_val_IoU_per_fold = [
            max(self.fold_metrics['val_IoU'][fold]) for fold in range(num_folds)
        ]


        train_mean = np.mean(best_train_dice_per_fold)
        train_std  = np.std(best_train_dice_per_fold)

        val_mean = np.mean(best_val_dice_per_fold)
        val_std  = np.std(best_val_dice_per_fold)

        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        for ax, data, title, color, mean, std in zip(
                axes,
                [best_train_dice_per_fold, best_val_dice_per_fold],
                ['Best Train Dice per Fold','Validation Dice per Fold'],
                ['orange','green'],
                [train_mean, val_mean],
                [train_std, val_std]):

            x = np.arange(num_folds)

            ax.scatter(x, data, color=color, s=80)

            offset = 0.001
            for i, v in enumerate(data):
                ax.text(i, v + offset, f"{v:.4f}", ha='center', fontsize=9)

            ax.axhline(mean, linestyle='--', color='red', label=f"Mean = {mean:.4f}")

            ax.set_title(f"{title}\nMean ± Std = {mean:.4f} ± {std:.4f}")

            ax.set_xlabel('Fold')
            ax.set_ylabel('Best Dice')
            ax.grid(True, axis='y')

            ax.set_ylim(min(data) - 0.002, max(data) + 0.003)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_dice_train_val.png"), dpi=300)
        plt.close()


        # -------------------------------
        # 5) Save μ ± σ summary for both
        # -------------------------------
        mu_train  = np.mean(best_train_dice_per_fold)
        sigma_train = np.std(best_train_dice_per_fold)
        mu_val    = np.mean(best_val_dice_per_fold)
        sigma_val   = np.std(best_val_dice_per_fold)
      
        IoU_mu_val    = np.mean(best_val_IoU_per_fold)
        IoU_sigma_val   = np.std(best_val_IoU_per_fold)
        

        with open(os.path.join(save_dir, "kfold_summary.txt"), "w") as f:
            f.write("==== K-Fold Summary ====\n\n")
            f.write(f"Number of folds: {num_folds}\n\n")
            f.write("Training Dice:\n")
            f.write(f"Best Dice per Fold: {best_train_dice_per_fold}\n")
            f.write(f"Mean (μ) = {mu_train:.4f}, Std (σ) = {sigma_train:.4f}\n")
            f.write(f"Result (μ ± σ) = {mu_train:.4f} ± {sigma_train:.4f}\n\n")
            f.write("Validation Dice:\n")
            f.write(f"Best Dice per Fold: {best_val_dice_per_fold}\n")
            f.write(f"Mean (μ) = {mu_val:.4f}, Std (σ) = {sigma_val:.4f}\n")
            f.write(f"Result (μ ± σ) = {mu_val:.4f} ± {sigma_val:.4f}\n\n")
            f.write("Validation IoU:\n")
            f.write(f"Best IoU per Fold: {best_val_IoU_per_fold}\n")
            f.write(f"Mean (μ) = {IoU_mu_val:.4f}, Std (σ) = {IoU_sigma_val:.4f}\n")
            f.write(f"Result (μ ± σ) = {IoU_mu_val:.4f} ± {IoU_sigma_val:.4f}\n")

        # -------------------------------
        # 6) Iteration Refinement Effect
        # -------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_final = np.mean(np.array(self.fold_metrics['train_dice']), axis=0)
        mean_all   = np.mean(np.array(self.fold_metrics['train_dice_all_iter']), axis=0)
        ax.plot(mean_final, label='Final Dice', color=color)
        ax.plot(mean_all, label='All-Iter Avg Dice', color="orange", linestyle='--')
        ax.fill_between(range(len(mean_final)), mean_all, mean_final, alpha=0.2, color="orange")
        ax.set_title('Train - Iterative Refinement Effect')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "iterative_refinement_effect_train.png"), dpi=300)
        plt.close()

        print(f"All K-Fold plots saved in: {save_dir}")
        print(f"Summary file saved as: {save_dir}/kfold_summary.txt")

if __name__ == "__main__":
    model_obj = run_train_cer()
    model_obj.load_data()
    model_obj.train_loop()
