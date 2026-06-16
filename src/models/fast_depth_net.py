import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Handle potential size mismatch (e.g., odd input sizes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FastDepthNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FastDepthNet, self).__init__()
        
        # ENCODER
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.enc1 = DepthwiseSepConv(32, 64)   # skip_1
        self.enc2 = DepthwiseSepConv(64, 128, stride=2)  # skip_2
        self.enc3 = DepthwiseSepConv(128, 256, stride=2) # skip_3
        self.enc4 = DepthwiseSepConv(256, 512, stride=2) 
        
        # DECODER
        self.dec3 = DecoderBlock(512, 256, skip_ch=256) # upsample enc4, cat enc3
        self.dec2 = DecoderBlock(256, 128, skip_ch=128) # upsample dec3, cat enc2
        self.dec1 = DecoderBlock(128, 64,  skip_ch=64)  # upsample dec2, cat enc1
        self.dec0 = DecoderBlock(64,  32,  skip_ch=0)   # upsample dec1
        
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)   # 32, 128, 128
        s1 = self.enc1(x0)  # 64, 128, 128
        s2 = self.enc2(s1)  # 128, 64, 64
        s3 = self.enc3(s2)  # 256, 32, 32
        x4 = self.enc4(s3)  # 512, 16, 16
        
        # Decoder
        d3 = self.dec3(x4, s3) # 256, 32, 32
        d2 = self.dec2(d3, s2) # 128, 64, 64
        d1 = self.dec1(d2, s1) # 64, 128, 128
        d0 = self.dec0(d1)     # 32, 256, 256
        
        return self.head(d0)

    def infer(self, image_bgr: np.ndarray, device="mps") -> np.ndarray:
        """
        Single image inference.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        img = cv2.resize(image_bgr, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            depth_rel = self.forward(input_tensor).squeeze().cpu().numpy()
            
        depth_rel = cv2.resize(depth_rel, (w_orig, h_orig))
        depth_metric = 10.0 * (1.0 - depth_rel)
        return depth_metric

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FastDepthNet().to(device)
    dummy = torch.randn(1, 3, 256, 256).to(device)
    out = model(dummy)
    print(f"Input: {dummy.shape}, Output: {out.shape}")
    print(f"Device: {device}")
