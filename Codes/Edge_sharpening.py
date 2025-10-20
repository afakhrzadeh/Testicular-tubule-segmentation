import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEdgeSobel:
    def __init__(self, device='cuda'):
        self.device = device
        # Define Sobel kernels in PyTorch
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        # Shape: [out_channels, in_channels, kernel_height, kernel_width]
        self.kernel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.kernel_y = sobel_y.view(1, 1, 3, 3).to(self.device)

    def rgb_to_gray(self, image):
        # Convert [1, 3, H, W] RGB image to grayscale [1, 1, H, W]
        if image.shape[1] == 3:
            r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = image  # already grayscale
        return gray

    def get_edge(self, image):
        # Ensure input is on the correct device
        image = image.to(self.device)
        gray = self.rgb_to_gray(image)  # [1, 1, H, W]

        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.kernel_x, padding=1)
        edge_y = F.conv2d(gray, self.kernel_y, padding=1)

        # Compute gradient magnitude
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)  # [1, 1, H, W]

        # Normalize to [0, 1]
        edge_map = edge_map / (edge_map.max() + 1e-8)

        # Resize to match SAM embedding resolution (64x64)
        edge_map_resized = F.interpolate(edge_map, size=(64, 64), mode='bilinear', align_corners=False)

        return edge_map_resized  # [1, 1, 64, 64]




class SimpleEdgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 512x512
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 256x256
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=2, padding=1),   # 64x64
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
