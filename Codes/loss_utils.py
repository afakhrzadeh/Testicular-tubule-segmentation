import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np

def calculate_loss(targets: torch.Tensor, pred: np.ndarray, sam_iou: np.ndarray, device, w_bce, w_focal, w_tversky, w_dice, w_iou):
    """Sam paper loss.

    Args:
        pred: prediction numpy array
        targets: ground truth Tensor
        sam_iou: model IOU score numpy array

    Returns:
        sam loss
    """
    # print(f"\nsam IOU {sam_iou} {type(sam_iou)}  {sam_iou.shape}")
    # Convert numpy arrays to PyTorch tensors
    
    # pred_tensor = torch.tensor(pred, dtype=torch.float32, requires_grad=True).to(device)
    # sam_iou_tensor = torch.tensor(sam_iou, dtype=torch.float32, requires_grad=True).to(device)
    
    # Calculate IOU
        
    # Compute loss
    # mse_loss = nn.MSELoss()(iou, sam_iou)  # Instantiate MSELoss and call it
    
    
    targets = targets.unsqueeze(0)

    bce_loss, focal_loss, dice_loss, tversky_loss, iou_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    if w_focal != 0:
        focal_loss = torch_focal_loss(targets, pred, gamma=4)
    if w_dice != 0:
        dice_loss = Dice_loss(targets, pred)
    if w_bce != 0:
        bce_loss = bce(targets, pred, device)
    if w_tversky != 0:
        tversky_loss = TverskyLoss(pred.unsqueeze(1) , targets.unsqueeze(1))
    if w_iou != 0:
        iou_loss = iou_pytorch(targets, pred)
 
    loss = w_bce * bce_loss + w_focal * focal_loss + w_tversky * tversky_loss + w_dice * dice_loss + w_iou * iou_loss

    return loss


def iou_pytorch(targets: torch.Tensor, pred: torch.Tensor, SMOOTH: float = 1e-6):
    """
    Computes IoU using TP, FP, and FN.
    
    Args:
        pred: Predicted logits or probabilities (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W)
        SMOOTH: Smoothing constant to avoid division by zero.

    Returns:
        Mean IoU score across the batch.
    """
    targets_binary = targets.float()
    
    pred = torch.sigmoid(pred)

    # Flatten tensors
    pred_flat = pred.view(pred.size(0), -1)
    targets_flat = targets_binary.view(targets_binary.size(0), -1)

    # Compute TP, FP, FN
    intersection = (pred_flat * targets_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return 1 - iou.mean()



def torch_focal_loss(targets: torch.Tensor, pred: torch.Tensor, alpha=0.25, gamma=2):
    """torch focal loss.

    Args:
        pred: prediction Tensor
        targets: ground truth Tensor

    Returns:
        focal loss

    """
    focal_loss = ops.sigmoid_focal_loss(
                                        inputs=pred, 
                                        targets=targets, 
                                        alpha=alpha, 
                                        gamma=gamma, 
                                        reduction='mean'
                                        )
    return focal_loss

def Dice_loss(targets: torch.Tensor, pred: torch.Tensor, smooth=1e-3):
    """Dice loss.

    Args:
        pred: prediction Tensor
        targets: ground truth Tensor

    Returns:
        1 - Dice loss

    """
    probs = torch.sigmoid(pred)
    # probs = F.softmax(pred, dim=1)
    targets = targets.float()

    # Flatten the tensors
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)

    # Compute the Dice coefficient
    intersection = (probs_flat * targets_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    # Compute Dice loss
    return 1 - dice_coeff

def bce(targets: torch.Tensor, pred: torch.Tensor, device):

    probs = torch.sigmoid(pred)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    bce_loss = loss_fn(pred, targets)
    return bce_loss

def TverskyLoss(preds, targets, alpha=0.7, beta=0.3, smooth=1e-6):

    preds = torch.sigmoid(preds)  # Convert logits to probabilities
    targets = targets.float()

    # Compute TP, FP, FN
    TP = (preds * targets).sum(dim=(2, 3))  # True Positives
    FP = ((1 - targets) * preds).sum(dim=(2, 3))  # False Positives
    FN = (targets * (1 - preds)).sum(dim=(2, 3))  # False Negatives

    # Compute Tversky Score
    tversky_score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    # Return Tversky Loss
    return 1 - tversky_score.mean()


def FocalTverskyLoss(preds, targets, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):

    preds = torch.sigmoid(preds)  # Convert logits to probabilities
    targets = targets.float()

    # Compute TP, FP, FN
    TP = (preds * targets).sum(dim=(2, 3))  # True Positives
    FP = ((1 - targets) * preds).sum(dim=(2, 3))  # False Positives
    FN = (targets * (1 - preds)).sum(dim=(2, 3))  # False Negatives

    # Compute Tversky Score
    tversky_score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    # Return Tversky Loss
    FTL = (1 - tversky_score) ** gamma
    return FTL.mean()


def Accuracy(binary_preds, targets):
    """(TP + TN) / (TP + FP + FN + TN)"""
    targets = targets.unsqueeze(0)
    targets = targets.float()
    tp = ((binary_preds != 0) & (targets != 0)).sum().item()
    tn = ((binary_preds == 0) & (targets == 0)).sum().item()
    fp = ((binary_preds != 0) & (targets == 0)).sum().item()
    fn = ((binary_preds == 0) & (targets != 0)).sum().item()
    # print(f"\n --> tp {tp} tn {tn} fp {fp} fn {fn} acc {float((tp + tn) / (tp + fp + tn + fn))}")
    # print(binary_preds)
    # print(targets)
    # assert 1 == 2
    return float((tp + tn) / (tp + fp + tn + fn))

def get_fp_fn(binary_preds, targets):

    def sample_points(fp, fn, num_points=512):
    
        fp_coords = torch.nonzero(fp, as_tuple=False)
        fn_coords = torch.nonzero(fn, as_tuple=False)

        
        all_coords = torch.cat([fp_coords, fn_coords], dim=0)
        
        total_points = all_coords.shape[0]
        if total_points == 0:
            return [], []
        
        if total_points > num_points:
            indices = torch.randperm(total_points)[:num_points]
            sampled = all_coords[indices]
        else:
            sampled = all_coords 

        points = []
        labels = []

        for coord in sampled:
            y, x = coord[1].item(), coord[2].item()
            points.append((int(x), int(y)))
            labels.append(0)
            
        return points, labels
    
    
    targets = targets.unsqueeze(0)
    targets = targets.float()
    
    binary_preds = binary_preds.bool()
    targets = targets.bool()
    
    fp = (binary_preds == 1) & (targets == 0)

    fn = (binary_preds == 0) & (targets == 1)

    # return fp, fn
    return sample_points(fp, fn)