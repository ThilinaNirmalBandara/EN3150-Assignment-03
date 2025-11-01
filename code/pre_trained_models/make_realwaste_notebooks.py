# make_realwaste_notebooks.py
# Creates four Jupyter notebooks that fine-tune popular ImageNet backbones on your
# RealWaste npy-splits (filepaths.npy, labels_encoded.npy, class_names.npy,
# split_train.npy, split_val.npy, split_test.npy, optional mean_std.npy).

try:
    import nbformat as nbf
except Exception:
    # Attempt to install nbformat at runtime if it's not available,
    # then import again. If installation fails, raise a helpful error.
    import subprocess, sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat"])
        import nbformat as nbf
    except Exception as e:
        raise ImportError("The 'nbformat' package is required but could not be imported or installed. "
                          "Please install it manually with: pip install nbformat") from e

from pathlib import Path

COMMON_HEADER = """# RealWaste — {model_name} (Using Manifest + Numpy Splits)

Loads dataset from your `.npy` splits (`filepaths.npy`, `labels_encoded.npy`, `class_names.npy`,
`split_train.npy`, `split_val.npy`, `split_test.npy`, optional `mean_std.npy`), fine-tunes
**{model_name}**, and reports Accuracy + macro Precision/Recall/F1 + confusion matrix.
"""

def build_notebook(model_name: str, model_block: str, img_size: int, out_subdir: str, extra_installs: str = ""):
    nb = nbf.v4.new_notebook()
    c = []

    # Intro
    c.append(nbf.v4.new_markdown_cell(COMMON_HEADER.format(model_name=model_name)))

    # Optional installs
    opt = "# (Optional) Installs (uncomment if needed)\n"
    if extra_installs:
        opt += extra_installs.strip() + "\n"
    opt += (
        "# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
        "# !pip install scikit-learn matplotlib pandas tqdm pillow numpy\n"
    )
    c.append(nbf.v4.new_code_cell(opt))

    # Paths / config
    c.append(nbf.v4.new_code_cell(f"""
from pathlib import Path
import numpy as np

SPLITS_DIR = Path("./")  # change if your npy files are elsewhere

FILEPATHS_NPY = SPLITS_DIR / "filepaths.npy"
LABELS_NPY    = SPLITS_DIR / "labels_encoded.npy"
CLASSES_NPY   = SPLITS_DIR / "class_names.npy"
TRAIN_NPY     = SPLITS_DIR / "split_train.npy"
VAL_NPY       = SPLITS_DIR / "split_val.npy"
TEST_NPY      = SPLITS_DIR / "split_test.npy"
MEAN_STD_NPY  = SPLITS_DIR / "mean_std.npy"

OUTPUT_DIR = Path("./outputs/{out_subdir}").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 133
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FT_EPOCHS = 25
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
IMG_SIZE = {img_size}
"""))

    # Repro + device
    c.append(nbf.v4.new_code_cell("""
import random, os, numpy as np, torch
def set_seed(seed=133):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
device
"""))

    # Load arrays + mean/std
    c.append(nbf.v4.new_code_cell("""
# Load arrays
filepaths = np.load(FILEPATHS_NPY, allow_pickle=True)
labels    = np.load(LABELS_NPY)
classes   = np.load(CLASSES_NPY, allow_pickle=True).tolist()
idx_tr    = np.load(TRAIN_NPY); idx_va = np.load(VAL_NPY); idx_te = np.load(TEST_NPY)

try:
    mean_std = np.load(MEAN_STD_NPY)
    mean = mean_std[0].tolist(); std = mean_std[1].tolist()
    print("Using dataset mean/std:", mean, std)
except Exception:
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    print("No mean_std.npy found; using ImageNet stats.")

len(classes), classes[:5], len(idx_tr), len(idx_va), len(idx_te)
"""))

    # Dataset + loaders
    c.append(nbf.v4.new_code_cell("""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NpySplitDataset(Dataset):
    def __init__(self, filepaths, labels, indices, img_size=224, train=True, mean=None, std=None):
        self.filepaths=filepaths; self.labels=labels; self.indices=indices; self.train=train
        self.mean = mean or [0.485,0.456,0.406]; self.std = std or [0.229,0.224,0.225]
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.15)), transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.1,0.1,0.1,0.05),
                transforms.ToTensor(), transforms.Normalize(self.mean,self.std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.15)), transforms.CenterCrop(img_size),
                transforms.ToTensor(), transforms.Normalize(self.mean,self.std),
            ])
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        idx=int(self.indices[i]); fp=str(self.filepaths[idx]); y=int(self.labels[idx])
        img=Image.open(fp).convert("RGB"); x=self.tf(img); return x,y

train_ds=NpySplitDataset(filepaths,labels,idx_tr,IMG_SIZE,True,mean,std)
val_ds  =NpySplitDataset(filepaths,labels,idx_va,IMG_SIZE,False,mean,std)
test_ds =NpySplitDataset(filepaths,labels,idx_te,IMG_SIZE,False,mean,std)

train_dl=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
val_dl  =DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)
test_dl =DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

len(train_ds), len(val_ds), len(test_ds)
"""))

    # Model-specific block (defines model + classifier_params)
    c.append(nbf.v4.new_code_cell(model_block))

    # Train/eval helpers
    c.append(nbf.v4.new_code_cell("""
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_one_epoch(model, dl, optimizer, device, smoothing=0.1):
    model.train(); total=correct=0; loss_sum=0.0
    for x,y in tqdm(dl, leave=False):
        x,y=x.to(device),y.to(device); optimizer.zero_grad()
        out=model(x); loss=F.cross_entropy(out,y,label_smoothing=smoothing); pred=out.argmax(1)
        loss.backward(); optimizer.step()
        bs=y.size(0); loss_sum+=loss.item()*bs; correct+=(pred==y).sum().item(); total+=bs
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); total=correct=0; loss_sum=0.0
    for x,y in dl:
        x,y=x.to(device),y.to(device); out=model(x); loss=F.cross_entropy(out,y); pred=out.argmax(1)
        bs=y.size(0); loss_sum+=loss.item()*bs; correct+=(pred==y).sum().item(); total+=bs
    return loss_sum/total, correct/total
"""))

    # Head training
    c.append(nbf.v4.new_code_cell(f"""
# 1) Train classifier head only
for p in model.parameters(): p.requires_grad=False
for p in classifier_params: p.requires_grad=True

opt=AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
sch=CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)

best_val_acc=-1.0; best_state=None
for ep in range(1, HEAD_EPOCHS+1):
    tr_loss,tr_acc=train_one_epoch(model, train_dl, opt, device, smoothing=LABEL_SMOOTH)
    va_loss,va_acc=evaluate(model, val_dl, device); sch.step()
    print(f"[head][{{ep:02d}}/{{HEAD_EPOCHS}}] train_acc={{tr_acc:.4f}}  val_acc={{va_acc:.4f}}")
    if va_acc>best_val_acc:
        best_val_acc=va_acc; best_state={{k:v.cpu() for k,v in model.state_dict().items()}}

from pathlib import Path; import torch
if best_state is not None:
    head_ckpt=Path(OUTPUT_DIR)/"{out_subdir.split('_')[0]}_head_best.pth"
    torch.save(best_state, head_ckpt); print("Saved:", head_ckpt)
"""))

    # Fine-tune all
    c.append(nbf.v4.new_code_cell(f"""
# 2) Fine-tune all layers
for p in model.parameters(): p.requires_grad=True
opt=AdamW(model.parameters(), lr=BASE_LR/3, weight_decay=WEIGHT_DECAY)
sch=CosineAnnealingLR(opt, T_max=FT_EPOCHS)

best_val_acc=-1.0; best_state=None
for ep in range(1, FT_EPOCHS+1):
    tr_loss,tr_acc=train_one_epoch(model, train_dl, opt, device, smoothing=LABEL_SMOOTH)
    va_loss,va_acc=evaluate(model, val_dl, device); sch.step()
    print(f"[ft  ][{{ep:02d}}/{{FT_EPOCHS}}] train_acc={{tr_acc:.4f}}  val_acc={{va_acc:.4f}}")
    if va_acc>best_val_acc:
        best_val_acc=va_acc; best_state={{k:v.cpu() for k,v in model.state_dict().items()}}

best_ckpt=Path(OUTPUT_DIR)/"{out_subdir.split('_')[0]}_best.pth"
if best_state is not None:
    torch.save(best_state, best_ckpt); print("Saved best fine-tuned checkpoint:", best_ckpt)
"""))

    # Final eval + CSV
    c.append(nbf.v4.new_code_cell(f"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd, numpy as np, torch

@torch.no_grad()
def predict_all(model, dl, device):
    model.eval(); ys=[]; ps=[]
    for x,y in dl:
        x=x.to(device); out=model(x); pred=out.argmax(1).cpu().numpy()
        ys.append(y.numpy()); ps.append(pred)
    y=np.concatenate(ys); p=np.concatenate(ps); return y,p

state=torch.load(best_ckpt, map_location="cpu"); model.load_state_dict(state)
test_loss,test_acc=evaluate(model, test_dl, device); y_true,y_pred=predict_all(model,test_dl,device)

acc=accuracy_score(y_true,y_pred)
prec,rec,f1,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
print("Test Accuracy:", acc); print("Macro Precision:", prec); print("Macro Recall:", rec); print("Macro F1:", f1)

df=pd.DataFrame([{{"model":"{out_subdir.split('_')[0]}","accuracy":acc,"precision_macro":prec,"recall_macro":rec,"f1_macro":f1}}])
csv_path=Path(OUTPUT_DIR)/"results.csv"; df.to_csv(csv_path,index=False); csv_path
"""))

    # Confusion matrix plot
    c.append(nbf.v4.new_code_cell(f"""
import matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,8)); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix — {model_name} (npy splits)')
plt.colorbar(); tick_marks=np.arange(len(classes)); plt.xticks(tick_marks, classes, rotation=90); plt.yticks(tick_marks, classes)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
cm_path=Path(OUTPUT_DIR)/"confusion_matrix_{out_subdir}.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight"); cm_path
"""))

    nb["cells"] = c
    return nb

# Model-specific code blocks (define `model` and `classifier_params`)
densenet_block = """
import torch.nn as nn
from torchvision import models
m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
m.classifier = nn.Linear(m.classifier.in_features, len(classes))
model = m.to(device)
classifier_params = list(model.classifier.parameters())
"""

vgg16bn_block = """
import torch.nn as nn
from torchvision import models
m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
m.classifier[6] = nn.Linear(m.classifier[6].in_features, len(classes))
model = m.to(device)
classifier_params = list(m.classifier[6].parameters())
"""

mobilenetv2_block = """
import torch.nn as nn
from torchvision import models
m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
model = m.to(device)
classifier_params = list(m.classifier[1].parameters())
"""

irv2_block = """
# Inception-ResNet-v2 via timm
# If timm isn't installed, uncomment:  # !pip install timm
import torch.nn as nn
try:
    import timm
except Exception as e:
    print("timm not installed. Install with: pip install timm")
    raise
m = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=len(classes))
model = m.to(device)
# timm sets the classifier with num_classes; we fine-tune it (and then all layers)
classifier_params = [p for p in model.parameters() if p.requires_grad]
"""

# Build and write files
targets = [
    ("DenseNet-121", densenet_block, 224, "densenet121_realwaste_splits", ""),
    ("VGG16-BN",     vgg16bn_block, 224, "vgg16bn_realwaste_splits", ""),
    ("MobileNetV2",  mobilenetv2_block, 224, "mobilenetv2_realwaste_splits", ""),
    ("Inception-ResNet-v2", irv2_block, 224, "inceptionresnetv2_realwaste_splits", "# !pip install timm"),
]

out_files = []
for name, block, size, subdir, extra in targets:
    nb = build_notebook(name, block, size, subdir, extra_installs=extra)
    out = Path(f"RealWaste_{subdir.replace('_realwaste_splits','')}_WithSplits.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    out_files.append(str(out.resolve()))

print("Wrote notebooks:")
for p in out_files:
    print(" -", p)
