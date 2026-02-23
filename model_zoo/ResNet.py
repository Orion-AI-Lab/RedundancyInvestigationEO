import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Deterministic pooling (replaces AdaptiveAvgPool2d for reproducibility)
# -------------------------
class _DeterministicGlobalAvgPool2d(nn.Module):
    """Deterministic replacement for AdaptiveAvgPool2d((1,1)); uses mean."""

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


# -------------------------
# ResNet18 Backbone (unchanged)
# -------------------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = _DeterministicGlobalAvgPool2d()
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Instead of global pooling here, we will use the spatial tokens for masking.
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

# -------------------------
# MaskedResNet18 with Top-k% Masking and Visualization
# -------------------------
class MaskedResNet18(nn.Module):
    def __init__(self, num_classes=100, mask_threshold=0.9, retain_percent=100, pad_value=0.0):
        """
        Args:
            num_classes: number of classification classes.
            retain_percent: percentage of tokens (patches) to keep based on cosine similarity.
            mask_threshold: cosine similarity threshold to decide token similarity.
            pad_value: value used for padding the token sequence.
        """
        super(MaskedResNet18, self).__init__()
        # Use the ResNet18 backbone up to layer4.
        self.backbone = ResNet18()
        # Remove the backbone’s avg_pool and fc to get the feature map.
        self.backbone.avg_pool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        # New classification head.
        self.classifier = nn.Linear(512, num_classes)
        
        # Masking parameters.
        self.retain_percent = retain_percent
        self.mask_threshold = mask_threshold
        self.pad_value = pad_value

    def batch_cos_similarity(self, x):
        """
        Compute cosine similarity among tokens for each sample.
        Args:
            x: tensor of shape (B, N, C)
        Returns:
            Tensor of shape (B, N, N)
        """
        norms = torch.norm(x, dim=2, keepdim=True)
        x_norm = x / (norms + 1e-8)
        cos_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        return cos_sim

    def get_mask_vectorised(self, cos_sim):
        """
        Vectorised implementation to get a boolean mask indicating tokens to keep.
        Args:
            cos_sim: tensor of shape (B, N, N)
        Returns:
            keep_mask: Boolean tensor of shape (B, N)
        """
        batch_size, num_tokens, _ = cos_sim.shape
        # Create an upper triangular mask to consider each pair only once.
        triu_mask = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
        sim_mask = cos_sim > self.mask_threshold
        combined_mask = sim_mask & triu_mask
        # For each token, if any token in front is too similar, mark it as not keep.
        mark_false = combined_mask.any(dim=1)  # (B, N)
        # Initially keep all tokens then remove those marked.
        keep_mask = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=cos_sim.device)
        keep_mask &= ~mark_false
        return keep_mask

    def get_topk_indices(self, cos_sim, k):
        """
        Get indices of top-k tokens based on their mean cosine similarity.
        Args:
            cos_sim: tensor of shape (B, N, N)
            k: number of tokens to retain.
        Returns:
            Indices tensor of shape (B, k)
        """
        batch_size, num_tokens, _ = cos_sim.shape
        mask = ~torch.eye(num_tokens, dtype=bool, device=cos_sim.device)
        masked_sim = cos_sim * mask.unsqueeze(0)
        mean_sim = masked_sim.sum(dim=2) / (num_tokens - 1)
        topk_indices = torch.topk(-mean_sim, k, dim=1).indices
        return topk_indices

    def calc_mask_vectorised(self, tokens, verbose=False):
        """
        Perform masking on the token embeddings.
        Args:
            tokens: Tensor of shape (B, N, C)
        Returns:
            Padded tensor of shape (B, topk_length, C) after masking.
        """
        B, N, C = tokens.shape
        cos_sim = self.batch_cos_similarity(tokens)
        keep_mask = self.get_mask_vectorised(cos_sim)
        if verbose:
            print("Keep mask for first sample:", keep_mask[0].tolist())
        
        topk_length = max(1, int(N * (self.retain_percent / 100.0)))
        topk_indices = self.get_topk_indices(cos_sim, topk_length)
        
        batch_indices = torch.arange(B, device=tokens.device)[:, None]
        masked_tokens = tokens[batch_indices, topk_indices]
        
        padded_tokens = torch.full((B, topk_length, C), self.pad_value, device=tokens.device)
        for i in range(B):
            token_count = masked_tokens[i].shape[0]
            padded_tokens[i, :token_count] = masked_tokens[i]
        
        if verbose:
            masked_ratio = 1 - (keep_mask.sum(dim=1).float() / N)
            print("Masked ratio (per sample):", masked_ratio.tolist())
        
        return padded_tokens, topk_indices, keep_mask

    def forward(self, x, verbose=False):
        # Pass through the backbone up to the final convolutional feature map.
        features = self.backbone.conv1(x)
        features = F.relu(self.backbone.bn1(features))
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        # features shape: (B, 512, H, W)
        
        B, C, H, W = features.shape
        tokens = features.view(B, C, H * W).transpose(1, 2)  # (B, N, C) with N = H*W
        
        masked_tokens, topk_indices, keep_mask = self.calc_mask_vectorised(tokens, verbose=verbose)
        pooled_features = masked_tokens.mean(dim=1)
        logits = self.classifier(pooled_features)
        return logits

    def visualize_mask(self, image, retain_percent=None, verbose=False):
        """
        Visualize an image before and after masking.
        Args:
            image: a single image tensor of shape (3, H, W) (unnormalized or normalized as desired).
            retain_percent: if provided, overrides the default retain_percent.
        """
        self.eval()
        if retain_percent is None:
            retain_percent = self.retain_percent
        
        # Add batch dimension.
        x = image.unsqueeze(0).to(next(self.parameters()).device)
        # Get backbone features.
        features = self.backbone.conv1(x)
        features = F.relu(self.backbone.bn1(features))
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        # features shape: (1, 512, H_f, W_f); for 32x32 input, expect H_f=W_f=4.
        B, C, H_f, W_f = features.shape
        tokens = features.view(B, C, H_f * W_f).transpose(1, 2)  # (1, N, C) with N = H_f*W_f
        
        # Use provided retain_percent.
        orig_retain = self.retain_percent
        self.retain_percent = retain_percent
        _, topk_indices, keep_mask = self.calc_mask_vectorised(tokens, verbose=verbose)
        self.retain_percent = orig_retain  # restore
        
        # Create a binary mask for tokens (shape: H_f*W_f)
        # Mark tokens kept by masking strategy as 1, others as 0.
        token_mask = torch.zeros(H_f * W_f, dtype=torch.float32, device=x.device)
        token_mask[topk_indices[0]] = 1.0  # for the single image
        
        # Reshape to feature map grid.
        token_mask = token_mask.view(H_f, W_f)
        # Upsample token mask to image size.
        # For 32x32 input and 4x4 feature map, each token corresponds to an 8x8 patch.
        patch_size = image.shape[1] // H_f  # assumes square image and even division
        mask_img = token_mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,H_f,W_f)
        mask_img = F.interpolate(mask_img, size=(image.shape[1], image.shape[2]), mode='nearest')
        mask_img = mask_img.squeeze().cpu().numpy()  # shape (H, W)
        
        # Create masked image: zero out pixels where mask is 0.
        image_np = image.cpu().permute(1, 2, 0).numpy()
        masked_image_np = image_np.copy()
        masked_image_np[mask_img < 0.5] = 0.0  # black out masked patches
        
        # Plot side by side.
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.clip(image_np, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(np.clip(masked_image_np, 0, 1))
        axes[1].set_title("Masked Image")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

# -------------------------
# Data Loading and Training Setup
# -------------------------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the masked model (adjust retain_percent, mask_threshold, pad_value as needed)
model = MaskedResNet18(num_classes=100, retain_percent=100, pad_value=0.0)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# -------------------------
# Training and Evaluation Loops (with Early Stopping)
# -------------------------
def train_model(model, trainloader, testloader, criterion, optimizer, epochs=20, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, verbose=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss, val_acc = evaluate_model(model, testloader, criterion, evaluate=True)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model:
        model.load_state_dict(best_model)

def evaluate_model(model, testloader, criterion, evaluate=False):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, verbose=False)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if evaluate:
        return test_loss / len(testloader), accuracy
    print(f'Test Loss: {test_loss/len(testloader):.4f}, Accuracy: {accuracy:.2f}%')

# Example: Visualize one image before and after masking.
# Get a sample from the testset (unnormalize for visualization if necessary)
sample_img, _ = testset[0]  # sample_img shape: (3, H, W)
# Optionally unnormalize if your transforms normalized the image.
# Here we assume image values are in [-1,1] after normalization, so convert to [0,1]:
sample_img_vis = (sample_img * 0.5) + 0.5

# Visualize the masking
model.visualize_mask(sample_img_vis, retain_percent=100, verbose=True)

# Train and evaluate the masked model
train_model(model, trainloader, testloader, criterion, optimizer)
evaluate_model(model, testloader, criterion)
