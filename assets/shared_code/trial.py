# improved_ms_gnn_fixed.py
# Requirements: python3.8+, torch, torchvision, numpy, pillow, scikit-image, scikit-learn, tqdm
# Usage: edit DATA_DIR and CACHE_DIR, then run: python improved_ms_gnn_fixed.py

import os, math, hashlib, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------
# Config / Hyperparams
# --------------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

IMG_SIZE = 224
PATCH_SIZE = 32
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
TEST_SPLIT = 0.2
DATA_DIR = r"C:\Users\gandh\OneDrive\Documents\patient_dataset_2d\MS"   # <-- set your dataset
CACHE_DIR = "./cache_ms_patches"
NUM_WORKERS = 0
USE_PRETRAINED = True
FORCE_REBUILD_CACHE = False
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 6

LABEL_MAP = {
    'Control Axial_crop': 0,
    'Control Saggital_crop': 0,
    'MS Axial_crop': 1,
    'MS Saggital_crop': 1
}

# Pretrained normalization (single-channel projection)
NORM_MEAN = 0.485
NORM_STD = 0.229

# --------------------
# Helpers
# --------------------
def makedirs(d): os.makedirs(d, exist_ok=True)
def file_hash(s): return hashlib.md5(s.encode('utf-8')).hexdigest()

# --------------------
# Image I/O + radiomics
# --------------------
def load_image_as_np(path, resize=(IMG_SIZE, IMG_SIZE), pad_to_patch=True, patch_size=PATCH_SIZE):
    img = Image.open(path).convert('L')
    img = img.resize(resize, Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = arr - arr.min()
    if pad_to_patch:
        H, W = arr.shape
        newH = int(math.ceil(H / patch_size) * patch_size)
        newW = int(math.ceil(W / patch_size) * patch_size)
        padH = newH - H
        padW = newW - W
        if padH > 0 or padW > 0:
            arr = np.pad(arr, ((0,padH),(0,padW)), mode='edge')
    return arr

def extract_radiomics(img_np):
    arr = (img_np*255).astype(np.uint8)
    mean,std = float(arr.mean()), float(arr.std())
    perc25, perc75 = float(np.percentile(arr,25)), float(np.percentile(arr,75))
    entr = float(shannon_entropy(arr))
    levels = 32
    arr_q = (arr / (256/levels)).astype(np.uint8)
    glcm = greycomatrix(arr_q, distances=[1], angles=[0, math.pi/4], levels=levels, symmetric=True, normed=True)
    contrast = float(greycoprops(glcm,'contrast').mean())
    dissimilarity = float(greycoprops(glcm,'dissimilarity').mean())
    homogeneity = float(greycoprops(glcm,'homogeneity').mean())
    energy = float(greycoprops(glcm,'energy').mean())
    correlation = float(greycoprops(glcm,'correlation').mean())
    vec = np.array([mean,std,perc25,perc75,entr,contrast,dissimilarity,homogeneity,energy,correlation],dtype=np.float32)
    return vec

# --------------------
# Patch graph building
# --------------------
def image_to_patches(img_np, patch_size=PATCH_SIZE):
    H,W = img_np.shape
    n_h, n_w = H//patch_size, W//patch_size
    patches = []
    coords = []
    for i in range(n_h):
        for j in range(n_w):
            patch = img_np[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
            coords.append((i,j))
    return np.stack(patches), coords, n_h, n_w

def build_patch_graph(img_np, cnn_model, device=DEVICE, patch_size=PATCH_SIZE):
    patches, coords, n_h, n_w = image_to_patches(img_np, patch_size=patch_size)
    N = patches.shape[0]
    with torch.no_grad():
        patch_t = torch.from_numpy(patches).unsqueeze(1).float().to(device)  # [N,1,H,W]
        patch_t = F.interpolate(patch_t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        # normalize to pretrained stats
        patch_t = (patch_t - NORM_MEAN) / NORM_STD
        feats = cnn_model(patch_t).cpu().numpy()
    rad_feats = np.array([extract_radiomics(p) for p in patches], dtype=np.float32)
    node_feats = np.concatenate([feats, rad_feats], axis=1).astype(np.float32)
    adj = np.zeros((N,N), dtype=np.float32)
    idx = lambda i,j: i*n_w + j
    for i,j in coords:
        u = idx(i,j)
        for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni,nj = i+di, j+dj
            if 0 <= ni < n_h and 0 <= nj < n_w:
                v = idx(ni,nj)
                adj[u,v] = 1.0
                adj[v,u] = 1.0
        adj[u,u] = 1.0
    return node_feats, adj, N

# --------------------
# CNN backbone
# --------------------
class CNNBackbone(nn.Module):
    def _init_(self, out_dim=128, pretrained=USE_PRETRAINED):
        super()._init_()
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained if hasattr(models.resnet18, '_call_') else False)
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        try:
            resnet.conv1.weight.data = w.mean(dim=1, keepdim=True)
        except Exception:
            pass
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(512, out_dim)
    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        feat = F.relu(self.fc(feat))
        return feat

# --------------------
# Simple GNN (mask-aware)
# --------------------
class SimpleGNN(nn.Module):
    def _init_(self, in_dim, hidden=64, out_dim=32, n_layers=2, dropout=0.2):
        super()._init_()
        self.linear_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, out_dim)
        self.n_layers = n_layers
        self.dropout = dropout
    def forward(self, X, A, mask=None):
        H = F.relu(self.linear_in(X))
        for i in range(self.n_layers):
            deg = A.sum(dim=-1, keepdim=True)
            A_norm = A / (deg + 1e-6)
            H_msg = torch.bmm(A_norm, H)
            B,N,Hdim = H_msg.shape
            H_lin = self.layers[i](H_msg)
            H_lin = self.bn[i](H_lin.view(B*N, Hdim)).view(B, N, Hdim)
            H = F.relu(H_lin)
            H = F.dropout(H, p=self.dropout, training=self.training)
        if mask is None:
            graph_feat = H.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).float()
            summed = (H * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            graph_feat = summed / denom
        out = self.readout(graph_feat)
        return out

# --------------------
# Full model
# --------------------
class CNN_Radiomics_GNN(nn.Module):
    def _init_(self, cnn_model, rad_len, gnn_hidden=64, gnn_out=32, num_classes=2):
        super()._init_()
        self.cnn = cnn_model
        node_in = self.cnn.fc.out_features + rad_len
        self.gnn = SimpleGNN(node_in, hidden=gnn_hidden, out_dim=gnn_out)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, node_feats, adj, mask=None):
        x = self.gnn(node_feats, adj, mask)
        logits = self.classifier(x)
        return logits

# --------------------
# Cache builder
# --------------------
def build_cache(samples, cnn_model, cache_dir=CACHE_DIR, force=False):
    makedirs(cache_dir)
    cnn_model.eval()
    built = 0
    for path,label in tqdm(samples, desc="Building cache", unit="sample"):
        key = file_hash(os.path.abspath(path))
        cache_file = os.path.join(cache_dir, f"{key}.npz")
        if os.path.exists(cache_file) and not force:
            continue
        img = load_image_as_np(path)
        node_feats, adj, N = build_patch_graph(img, cnn_model, device=DEVICE)
        np.savez_compressed(cache_file,
                            node_feats=node_feats.astype(np.float32),
                            adj=adj.astype(np.float32),
                            n_nodes=np.int32(N),
                            label=np.int32(label))
        built += 1
    print(f"Cache complete. New files: {built}")

# --------------------
# Dataset & collate
# --------------------
class PatchGraphDataset(Dataset):
    def _init_(self, samples, cache_dir=CACHE_DIR):
        self.samples = samples
        self.cache_dir = cache_dir
        missing = 0
        for path,_ in samples:
            key = file_hash(os.path.abspath(path))
            if not os.path.exists(os.path.join(cache_dir, f"{key}.npz")):
                missing += 1
        if missing:
            raise RuntimeError(f"{missing} cache entries missing. Run build_cache(...) first.")
        self.index = [os.path.join(cache_dir, f"{file_hash(os.path.abspath(p))}.npz") for p,_ in samples]
    def _len_(self): return len(self.index)
    def _getitem_(self, idx):
        arr = np.load(self.index[idx])
        node_feats = torch.from_numpy(arr['node_feats']).float()
        adj = torch.from_numpy(arr['adj']).float()
        n_nodes = int(arr['n_nodes'])
        label = int(arr['label'])
        mask = torch.ones(n_nodes, dtype=torch.bool)
        return node_feats, adj, mask, torch.tensor(label, dtype=torch.long)

def collate_pad(batch):
    maxN = max(x[0].shape[0] for x in batch)
    Fdim = batch[0][0].shape[1]
    B = len(batch)
    node_batch = torch.zeros(B, maxN, Fdim, dtype=torch.float32)
    adj_batch = torch.zeros(B, maxN, maxN, dtype=torch.float32)
    mask_batch = torch.zeros(B, maxN, dtype=torch.bool)
    label_batch = torch.zeros(B, dtype=torch.long)
    for i,(nf,a,m,l) in enumerate(batch):
        N = nf.shape[0]
        node_batch[i,:N,:] = nf
        adj_batch[i,:N,:N] = a
        mask_batch[i,:N] = m
        label_batch[i] = l
    return node_batch, adj_batch, mask_batch, label_batch

# --------------------
# Metrics & predict
# --------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def predict_image(path, cnn_model, model, device=DEVICE, patch_size=PATCH_SIZE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    model.eval()
    img = load_image_as_np(path)
    node_feats, adj, N = build_patch_graph(img, cnn_model, device=device, patch_size=patch_size)
    node_t = torch.from_numpy(node_feats).unsqueeze(0).to(device)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(device)
    mask_t = torch.ones(1, N).to(device)
    with torch.no_grad():
        logits = model(node_t, adj_t, mask_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    label_name = "MS" if pred == 1 else "Control"
    return label_name, probs

# --------------------
# Main flow
# --------------------
if _name_ == "_main_":
    # 1) gather samples
    torch.set_num_threads(1)
    samples = []
    for cls_folder in os.listdir(DATA_DIR):
        full_cls = os.path.join(DATA_DIR, cls_folder)
        if os.path.isdir(full_cls) and cls_folder in LABEL_MAP:
            label = LABEL_MAP[cls_folder]
            for f in os.listdir(full_cls):
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    samples.append((os.path.join(full_cls, f), label))
    print("Total samples:", len(samples))
    if len(samples) < 4:
        raise RuntimeError("Not enough samples found. Check DATA_DIR and LABEL_MAP.")

    # 2) split
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SPLIT,
                                                  stratify=[l for _,l in samples], random_state=SEED)
    print("Train:", len(train_samples), "Val:", len(val_samples))

    # 3) encoder
    cnn_encoder = CNNBackbone(out_dim=128, pretrained=USE_PRETRAINED).to(DEVICE)
    cnn_encoder.eval()

    # 4) cache (heavy)
    build_cache(train_samples + val_samples, cnn_encoder, cache_dir=CACHE_DIR, force=FORCE_REBUILD_CACHE)

    # 5) datasets + loaders
    train_ds = PatchGraphDataset(train_samples, cache_dir=CACHE_DIR)
    val_ds = PatchGraphDataset(val_samples, cache_dir=CACHE_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad, num_workers=NUM_WORKERS)

    # 6) model
    # read rad_len from a cache file (robust)
    any_cache = next(Path(CACHE_DIR).glob("*.npz"))
    node_feats_shape = np.load(str(any_cache))['node_feats'].shape
    rad_len_example = node_feats_shape[1] - cnn_encoder.fc.out_features
    model = CNN_Radiomics_GNN(cnn_encoder, rad_len=rad_len_example, gnn_hidden=64, gnn_out=32, num_classes=2).to(DEVICE)

    # 7) class weights & loss
    labels_train = [l for _,l in train_samples]
    counts = np.bincount(labels_train, minlength=2).astype(np.float32)
    class_weights = torch.tensor((len(labels_train) / (counts + 1e-6))).float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # FIX: remove verbose (older torch versions)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 8) training loop with early stopping
    best_val = 0.0
    patience_counter = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train", leave=False)
        for node_feats, adj, mask, labels in pbar:
            node_feats = node_feats.to(DEVICE); adj = adj.to(DEVICE); mask = mask.to(DEVICE); labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(node_feats, adj, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred_train.extend(preds.tolist())
            y_true_train.extend(labels.detach().cpu().numpy().tolist())
            pbar.set_postfix({"loss": running_loss / (len(y_true_train) + 1e-12)})
        train_metrics = compute_metrics(y_true_train, y_pred_train)

        # validation
        model.eval()
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for node_feats, adj, mask, labels in val_loader:
                node_feats = node_feats.to(DEVICE); adj = adj.to(DEVICE); mask = mask.to(DEVICE)
                logits = model(node_feats, adj, mask)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred_val.extend(preds.tolist())
                y_true_val.extend(labels.numpy().tolist())
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        scheduler.step(val_metrics["accuracy"])
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Loss: {running_loss / len(train_ds):.4f}")
        print(f"  Val precision: {val_metrics['precision']:.4f} recall: {val_metrics['recall']:.4f} f1: {val_metrics['f1']:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_metrics["accuracy"] > best_val + 1e-6:
            best_val = val_metrics["accuracy"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "cnn_state": cnn_encoder.state_dict(),
                "rad_len": rad_len_example,
                "label_map": LABEL_MAP
            }, "best_model.pth")
            print("Saved best model. Val Acc:", best_val)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
                break

    print("Best validation accuracy:", best_val)

    # quick test on a val sample
    sample_path = val_samples[0][0]
    label_name, probs = predict_image(sample_path, cnn_encoder, model, device=DEVICE, patch_size=PATCH_SIZE)
    print("Sample:", sample_path, "True:", val_samples[0][1], "Pred:", label_name, "Probs:", probs)