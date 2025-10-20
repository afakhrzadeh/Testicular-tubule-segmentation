import torch
import torch.nn as nn
from torchvision import transforms
  
class Resize_model(nn.Module):
# Resizeing images and masks. 
# Resizing from any size to 1024 for sam input.
# Also resizing to 224 for UNI encoder. 
    def __init__(self) -> None:
        super().__init__()

        self.resize_to_1024 = transforms.Resize((1024, 1024))
        self.resize_to_224 = transforms.Resize((224,224))
    
    def forward(self, x) -> torch.tensor:
        # x :  demonstration with a batch of images in the format (B, C, H, W)
        with torch.no_grad():
            x_1024 = self.resize_to_1024(x)
            x_224 = self.resize_to_224(x)

        return x_1024, x_224
    
    

class Resize_model_for_mask(nn.Module):
# Resizeing images and masks. 
# Resizing from any size to 1024 for sam input.
# Also resizing to 224 for UNI encoder. 
    def __init__(self) -> None:
        super().__init__()

        self.resize_to_1024 = transforms.Resize((1024//4, 1024//4))
    
    def forward(self, x) -> torch.tensor:
        # x :  demonstration with a batch of images in the format (B, C, H, W)
        with torch.no_grad():
            x_1024 = self.resize_to_1024(x)

        return x_1024