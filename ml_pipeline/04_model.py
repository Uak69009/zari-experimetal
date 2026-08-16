"""
ZARI.ai Neural Network Architecture (Phase 4 - Implementation)
--------------------------------------------------------------
Model: EfficientNetV2-S with Coordinate Attention & Evidential Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    """
    Coordinate Attention Module.
    Encodes spatial information into the channel dimension to help the network
    focus on fine-grained disease lesions rather than random background elements.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ZariNet(nn.Module):
    """
    ZARI.ai Main Architecture.
    - Backbone: EfficientNetV2-S
    - Attention: CoordAtt applied to the final feature map.
    - Evidential Head: Outputs Dirichlet Evidence (>0).
    """
    def __init__(self, num_classes=153, backbone_name='tf_efficientnetv2_s.in21k_ft_in1k', embed_dim=256):
        super(ZariNet, self).__init__()
        
        # 1. Backbone (strip head and global pool)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='')
        
        # Automatically determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
            
        # 2. Coordinate Attention
        self.coord_att = CoordAtt(inp=feature_dim, oup=feature_dim)
        
        # 3. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Shared Embedding Space
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        # 5. Evidential Head (Dirichlet Alpha = Evidence + 1)
        self.fc_evidential = nn.Linear(embed_dim, num_classes)
        self.activation = nn.Softplus() # Ensures strictly positive evidence

    def forward(self, x):
        # Backbone Features
        x = self.backbone(x)
        
        # Inject Coordinate Attention
        x = self.coord_att(x)
        
        # Pool
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Shared Embedding
        emb = self.embedding(x)
        
        # Evidential Output
        logits = self.fc_evidential(emb)
        
        # Softplus ensures the output (evidence) is strictly positive > 0
        evidence = self.activation(logits)
        
        return evidence, emb

if __name__ == "__main__":
    print("Testing ZariNet Architecture mathematically...")
    
    # 153 ZARI classes
    model = ZariNet(num_classes=153)
    model.eval()
    
    # Dummy Batch: 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    print(f"Input Shape: {dummy_input.shape}")
    
    with torch.no_grad():
        evidence, embedding = model(dummy_input)
    
    print(f"Output Evidence Shape: {evidence.shape} (Expected: 4, 153)")
    print(f"Output Embedding Shape: {embedding.shape} (Expected: 4, 256)")
    
    # Verify strict positivity
    min_evidence = evidence.min().item()
    print(f"Minimum Evidence Value: {min_evidence:.4f} (Must be > 0)")
    
    if min_evidence >= 0 and evidence.shape == (4, 153):
        print("\n[OK] ZariNet Architecture Passes Verification!")
    else:
        print("\n[ERROR] ZariNet Architecture Verification Failed.")
