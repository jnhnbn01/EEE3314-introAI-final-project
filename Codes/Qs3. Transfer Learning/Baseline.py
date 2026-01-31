import torch
from torch import nn, optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import math, copy

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# ============================== Network Table ==============================
def fmapsize(i, s, p, k):
    return math.floor((i + 2*p - k)/s) + 1

def print_architecture(model):
    layers = {
        'conv1':'Conv1', 'maxpool':'Pooling',
        'layer1[0].conv1':'Conv2-1', 'layer1[0].conv2':'Conv2-2',
        'layer1[1].conv1':'Conv2-3', 'layer1[1].conv2':'Conv2-4',
        'layer2[0].conv1':'Conv3-1', 'layer2[0].conv2':'Conv3-2',
        'layer2[1].conv1':'Conv3-3', 'layer2[1].conv2':'Conv3-4',
        'layer3[0].conv1':'Conv4-1', 'layer3[0].conv2':'Conv4-2',
        'layer3[1].conv1':'Conv4-3', 'layer3[1].conv2':'Conv4-4',
        'layer4[0].conv1':'Conv5-1', 'layer4[0].conv2':'Conv5-2',
        'layer4[1].conv1':'Conv5-3', 'layer4[1].conv2':'Conv5-4',
        'avgpool': 'avgpool'
    }
    print(f"{'Layer':<10}{'Stride':<7}{'Padding':<8}{'Fmap Size':<10}{'# of weights':<13}{'Receptive Field':<16}")
    fmap_size = [64, 3]
    R = 1
    prod_s = 1
    size = str(fmap_size[0])+'x'+str(fmap_size[0])+'x'+str(fmap_size[1])
    print(f"{'Input':<10}{'-':<7}{'-':<8}{size:<10}{'-':<13}{1:<16}")

    
    for i in layers:
        layer = eval(f"model.{i}")
        if i == 'maxpool':
            s, p, k, out_c, num_w = layer.stride, layer.padding, layer.kernel_size, out_c, 0
            fmap_size = [fmapsize(fmap_size[0], s, p, k), out_c]
        elif i == 'avgpool':
            (h, w), num_w = layer.output_size, 0
            fmap_size = [h, out_c]
        else:
            s, p, k, out_c, num_w = layer.stride[0], layer.padding[0], layer.kernel_size[0], layer.out_channels, layer.weight.numel()
            if(i != 'conv1' and i != 'layer1[0].conv1' and i[-1] == '1' and i[-8] == '0'): # add skip connection's params to conv()_1 layer
                skip_layer = eval(f"model.{i.rsplit('.', 1)[0]}.downsample")
                num_w += skip_layer[0].weight.numel() # Conv_skip
                #num_w += skip_layer[1].weight.numel() # BN_skip gamma
                #num_w += skip_layer[1].bias.numel() # BN_skip beta
            #if(i == 'conv1'): # Including BN layers' parameters
            #    num_w += eval(f"model.bn1.weight.numel()") # gamma
            #    num_w += eval(f"model.bn1.bias.numel()") # beta
            #else:
            #    num_w += eval(f"model.{i.rsplit('.', 1)[0]}.bn{i[-1]}.weight.numel()") # gamma
            #    num_w += eval(f"model.{i.rsplit('.', 1)[0]}.bn{i[-1]}.bias.numel()") # beta
            
            fmap_size = [fmapsize(fmap_size[0], s, p, k), out_c]
        
        if i == 'conv1':
            RF = k
            prod_s *= s
        elif i == 'avgpool':
            None
        else:
            RF += (k - 1)*prod_s
            prod_s *= s
        size = str(fmap_size[0])+'x'+str(fmap_size[0])+'x'+str(fmap_size[1])
        if i == 'avgpool':
            print(f"{layers.get(i):<10}{'-':<7}{'-':<8}{size:<10}{num_w:<13}{'-':<16}")
        else:
            print(f"{layers.get(i):<10}{s:<7}{p:<8}{size:<10}{num_w:<13}{RF:<16}")


# ============================== Modified ResNet18 ==============================
print("Original ResNet18 Total # of Weights(w/o FC head):", sum(p.numel() for p in model.parameters() if p.requires_grad) - 513000) # FC head has 512*1000 + 1000 = 513000 parameters

model.conv1.stride = (1, 1)
model.maxpool.stride = 1

print("Modified ResNet18 Total # of Weights(w/o FC head):", sum(p.numel() for p in model.parameters() if p.requires_grad) - 513000) # FC head has 512*1000 + 1000 = 513000 parameters
print_architecture(model) # The shape of output fmap of conv5 will be 8x8x512


# ============================== Dataset Construction ==============================
train_path = str(Path(__file__).resolve().parent) + '/face_dataset/facescrub_train'
test_path = str(Path(__file__).resolve().parent) + '/face_dataset/facescrub_test'

def load_face_dataset(train_dir, test_dir, batch_size=32):
    """
    Load face dataset with proper preprocessing
    """
    # ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # No data augmentation except for normalization for Baseline model
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    num_classes = len(train_dataset.classes)
    
    n_total = len(train_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val]) # Split into train set & validation set
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, num_classes

# ============================== Training & Testing Utils ==============================

def Train(model, tr_loader, criterion, optimizer):
    NoT = len(tr_loader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0
    accuracy = 0
    for x_batch, y_batch in tr_loader:
        x_batch = x_batch.to(device) # Move to GPU
        y_batch = y_batch.to(device)
        optimizer.zero_grad() # Gradient Initialize
        y_hat = model(x_batch) # Inference
        loss = criterion(y_hat, y_batch) # Loss Calculation
        loss.backward() # Backward Propagation
        optimizer.step() # Update
        
        loss_batch = loss.item() * x_batch.shape[0]
        running_loss += loss_batch
        pred = torch.argmax(y_hat, dim = 1)
        correct += torch.sum(y_batch == pred).item()
    loss = running_loss / NoT
    accuracy = correct / NoT
    return loss, accuracy

def Test(model, te_loader, criterion):
    NoT = len(te_loader.dataset)
    model.eval()
    with torch.no_grad():
        running_loss = 0
        correct = 0
        accuracy = 0
        for x_batch, y_batch in te_loader:
            x_batch = x_batch.to(device) # Move to GPU
            y_batch = y_batch.to(device)
            y_hat = model(x_batch) # Inference
            loss = criterion(y_hat, y_batch)
            loss_batch = loss.item() * x_batch.shape[0]
            running_loss += loss_batch
            pred = torch.argmax(y_hat, dim = 1) # Inference result to class index
            correct += torch.sum(y_batch == pred).item() # Count # of correct predictions
        loss = running_loss / NoT
        accuracy = correct / NoT
    return loss, accuracy

def plot_loss_history(train_loss_history, val_loss_history, test_loss_history, train_accs, val_accs, test_accs, plot_save_path, model_name):
    """Plot training, test history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_loss_history) + 1)
    # Loss plot
    ax1.plot(epochs, train_loss_history, label='Train Loss')
    ax1.plot(epochs, val_loss_history, label='Validation Loss')
    ax1.plot(epochs, test_loss_history, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, np.array(train_accs) * 100, label='Train Accuracy')
    ax2.plot(epochs, np.array(val_accs) * 100, label='Validation Accuracy')
    ax2.plot(epochs, np.array(test_accs) * 100, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

# ============================== Transfer Learning ==============================

train_flag = False  # If you want train the models, set train_flag to True
param_choice = False # flag for choosing parameters

# --------------------------- Construct the baseline model ---------------------------
# Change the FC layer
model.fc = nn.Linear(model.fc.in_features, 100, bias=True) 

# Freezing the other layers in the network except for only a linear classifier.
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()

# --------------------------- Find the optimal parameters ---------------------------
EPOCH = 40
best_val_acc = 0.0
if train_flag and param_choice:
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]: # Grid search to find optimal params
        for b in [8, 16, 32, 64, 128]:
            tr_loader, val_loader, _, num_classes = load_face_dataset(train_path, test_path, batch_size=b) # train, val data loader
            
            fresh_model = copy.deepcopy(model) # reload the model
            fresh_model = fresh_model.to(device)
            
            optimizer = optim.Adam(fresh_model.fc.parameters(), lr = lr)
            # Step LR scheduler: reduce LR by a factor of 10 at 1/3 and 2/3 of total epochs
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
            
            running_acc = 0.0
            running_epoch = 0
            print(f"========================== LR: {lr} / BATCH: {b} ==========================")
            for epoch in range(EPOCH):
                tr_loss, tr_acc = Train(fresh_model, tr_loader, criterion, optimizer)
                val_loss, val_acc = Test(fresh_model, val_loader, criterion)
                scheduler.step(val_loss)
                print(f"Epoch {epoch+1}: Train Loss {round(tr_loss, 3)} / Train Acc {round(tr_acc, 3)} / Val Loss {round(val_loss, 3)} / Val Acc {round(val_acc, 3)}")
                if val_acc > running_acc: # record best val acc of each parameter settting
                    running_acc, running_epoch = val_acc, epoch  
            
            print(f"Best Val Accuracy: {running_acc} / Epoch: {running_epoch + 1}")
            if running_acc > best_val_acc: # record best val acc ammong all parameter setttings
                best_val_acc, best_lr, best_b = running_acc, lr, b

    print(f"Best Batch Size: {best_b}, Best Learning Rate: {best_lr}, Best Val Acc: {best_val_acc}")


# --------------------------- Train the model with the optimal parameters ---------------------------
BATCH_SIZE = 8
LR = 5e-3
EPOCH = 40
if train_flag and not param_choice:
    tr_loader, val_loader, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE)
    model = model.to(device)
    save_model_path = str(Path(__file__).resolve().parent) + "/baseline.pt"
    optimizer = optim.Adam(model.fc.parameters(), lr = LR)
    # Step LR scheduler: reduce LR by a factor of 10 at 1/3 and 2/3 of total epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    
    # Loss & Acc History for Train, Val, Test Dataset
    tr_loss_history, tr_acc_history = [], []
    val_loss_history, val_acc_history = [], []
    te_loss_history, te_acc_history = [], []
    
    best_te_acc = 0.0
    best_te_loss = 0.0
    best_epoch = 0
    for epoch in range(EPOCH):
        tr_loss, tr_acc = Train(model, tr_loader, criterion, optimizer)
        val_loss, val_acc = Test(model, val_loader, criterion)
        te_loss, te_acc = Test(model, te_loader, criterion)
        scheduler.step(val_loss)
        
        tr_loss_history.append(tr_loss)
        tr_acc_history.append(tr_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        te_loss_history.append(te_loss)
        te_acc_history.append(te_acc)
        
        print(f"Epoch {epoch+1} = Train Loss: {round(tr_loss, 3)}, Train Acc: {round(tr_acc*100, 3)}% / "
        f"Val Loss: {round(val_loss, 3)}, Val Acc: {round(val_acc*100, 3)}% / "
        f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")
        if best_te_acc < te_acc_history[-1]: # find best test accuracy
            best_te_acc = te_acc_history[-1]
            best_te_loss = te_loss_history[-1]
            best_epoch = epoch
    torch.save(model, save_model_path)
    plot_loss_history(tr_loss_history, val_loss_history, te_loss_history, tr_acc_history, val_acc_history, te_acc_history, str(Path(__file__).resolve().parent) + '/baseline.png', f"Baseline(LR = {LR}, Batch Size = {BATCH_SIZE})")
    print(f"Best: Epoch {best_epoch + 1} / Test Acc: {round(best_te_acc * 100, 3)}% / Test Loss: {round(best_te_loss, 3)}")
elif not train_flag and not param_choice: 
    pretrained_model_path = str(Path(__file__).resolve().parent) + "/baseline.pt"
    _, _, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE)
    model = torch.load(pretrained_model_path, map_location=device, weights_only=False) # Load the pretrained model
    te_loss, te_acc = Test(model, te_loader, criterion)
    print(f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")