"""ResNet-style backbone + YOLO-style detection head, trained from scratch."""
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_ANCHORS, NUM_CLASSES


class ResBlock(nn.Module):
    """
    Standard residual block: Conv-BN-LReLU → Conv-BN → skip add → LReLU.
    A 1×1 projection is used when channel count or stride changes.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + self.skip(x), 0.1)


class ScratchDetectorBackbone(nn.Module):
    """
    Five-stage CNN backbone.
    Input  : (B, 3,   416, 416)
    Output : (B, 512,  13,  13)

    Stem   : 3→64,   7×7 conv, stride 2 + MaxPool  → 104×104
    Stage1 : 64→128,  stride 2  (1 ResBlock)        →  52×52
    Stage2 : 128→256, stride 2  (2 ResBlocks)       →  26×26
    Stage3 : 256→512, stride 2  (2 ResBlocks)       →  13×13
    Stage4 : 512→512, stride 1  (2 ResBlocks)       →  13×13  (refine)
    """
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._stage(64,  128, n=1, stride=2)
        self.stage2 = self._stage(128, 256, n=2, stride=2)
        self.stage3 = self._stage(256, 512, n=2, stride=2)
        self.stage4 = self._stage(512, 512, n=2, stride=1)

    @staticmethod
    def _stage(in_ch, out_ch, n, stride):
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n - 1):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x   # (B, 512, 13, 13)


class ScratchDetectorHead(nn.Module):
    """
    Detection head: maps (B, 512, 13, 13) → (B, G, G, A, 5+C).

    Per anchor output:
        [..., 0]   : raw objectness logit  (σ → P(object))
        [..., 1:3] : (tx, ty)  centre offset within grid cell  (σ applied at decode)
        [..., 3:5] : (tw, th)  log-scale correction relative to anchor size
        [..., 5:]  : class logits  (C values)
    """
    def __init__(self):
        super().__init__()
        mid = 256
        self.conv1  = nn.Conv2d(512, mid, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(mid)
        self.conv2  = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(mid)
        self.output = nn.Conv2d(mid, NUM_ANCHORS * (5 + NUM_CLASSES), 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = self.output(x)                          # (B, A*(5+C), G, G)
        B, _, G, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()     # (B, G, G, A*(5+C))
        x = x.view(B, G, G, NUM_ANCHORS, 5 + NUM_CLASSES)
        return x


class ScratchDetector(nn.Module):
    """Complete detector: backbone + head."""
    def __init__(self):
        super().__init__()
        self.backbone = ScratchDetectorBackbone()
        self.head     = ScratchDetectorHead()

    def forward(self, x):
        return self.head(self.backbone(x))
