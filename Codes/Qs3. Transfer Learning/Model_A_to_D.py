import torch
from torch import nn, optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import copy 

# ============================== Dataset Construction ==============================
train_path = str(Path(__file__).resolve().parent) + '/face_dataset/facescrub_train'
test_path = str(Path(__file__).resolve().parent) + '/face_dataset/facescrub_test'

def load_face_dataset(train_dir, test_dir, batch_size=32, data_aug=1):
    """
    Load face dataset with proper preprocessing
    """
    # ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if data_aug == 1: # Data augmentation for Model A, B
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), # Random Horizontal Flip with 50% probability
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9, 1.1)), # Random Affine Transformation(rotation degree=15, translation up to 10%, scaling between 90% to 110%)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Random Color Jitter(Brightness, Contrast, Saturation changes up to 20%)
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 2: # Data augmentation for Model C; adding more augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9, 1.1)), # Random Affine Transformation(rotation degree=15, translation up to 10%, scaling between 90% to 110%)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Random Color Jitter(Brightness, Contrast, Saturation changes up to 30%)
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([ # Data augmentation for Model D(Remove RandomAffine because mlp cannot have inductive biases of CNN)
            transforms.RandomHorizontalFlip(p=0.5), # Random Horizontal Flip with 50% probability
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Random Color Jitter(Brightness, Contrast, Saturation changes up to 20%)
            transforms.ToTensor(),
            normalize
        ])
    
    
    # Test transform - no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    
    
    learning_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())  # no aug
    num_classes = len(learning_dataset.classes)
    indices = torch.randperm(len(learning_dataset))
    n_total = len(learning_dataset)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=train_dir, transform=train_transform),
        train_idx
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=train_dir, transform=test_transform),
        val_idx
    )
    '''
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    num_classes = len(train_dataset.classes)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val]) # Split into train set & validation set
    val_dataset.transform = test_transform # using transform of test dataset for validation dataset
    '''
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
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

save_path = str(Path(__file__).resolve().parent) # Path where the model will be saved

def train_model_a(train_flag, save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE):
    tr_loader, val_loader, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE) # Data Loadder
    save_model_path = save_path + "/model_a.pt"
    
    # Finetuning
    finetuned_model = copy.deepcopy(baseline)
    for param in finetuned_model.parameters(): # Freeze All Parameters
        param.requires_grad = False
    for param in finetuned_model.fc.parameters(): # Unfreeze FC Layer
        param.requires_grad = True
    for param in finetuned_model.layer4.parameters(): # Unfreeze Conv5_x
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss() # Loss

    if train_flag: # Retrain the model
        finetuned_model = finetuned_model.to(device)
        optimizer = torch.optim.Adam([
            {"params": finetuned_model.fc.parameters(),     "lr": LR/10},  # head 1/10
            {"params": finetuned_model.layer4.parameters(), "lr": LR/10},  # late conv 1/10
        ], weight_decay=0.0001)
        # LR scheduler: reduce LR by a factor of 10 at plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
        
        # Loss & Acc History for Train, Val, Test Dataset
        tr_loss_history, tr_acc_history = [], []
        val_loss_history, val_acc_history = [], []
        te_loss_history, te_acc_history = [], []
        
        best_te_acc = 0.0
        best_te_loss = 0.0
        best_epoch = 0
        for epoch in range(EPOCH):
            tr_loss, tr_acc = Train(finetuned_model, tr_loader, criterion, optimizer)
            val_loss, val_acc = Test(finetuned_model, val_loader, criterion)
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
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
                
        torch.save(finetuned_model, save_model_path)
        plot_loss_history(tr_loss_history, val_loss_history, te_loss_history, tr_acc_history, val_acc_history, te_acc_history, str(Path(__file__).resolve().parent) + '/model_a.png', f"Model A(LR = {LR}, Batch Size = {BATCH_SIZE})")
        print(f"Best: Epoch {best_epoch + 1} / Test Acc: {round(best_te_acc * 100, 3)}% / Test Loss: {round(best_te_loss, 3)}")

    else: 
        try:
            finetuned_model = torch.load(save_model_path, map_location=device, weights_only=False) # Load the pretrained model
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
            print(f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")
        except FileNotFoundError:
            print(f"Pre-trained model doesn't exist at {save_model_path}")


def train_model_b(train_flag, save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE):
    save_model_path = save_path + "/model_b.pt"
        # Data Loadder
    tr_loader, val_loader, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE)

    # Finetuning
    finetuned_model = copy.deepcopy(baseline)
    for param in finetuned_model.parameters(): # Freeze All Parameters
        param.requires_grad = False
    for param in finetuned_model.fc.parameters(): # Unfreeze FC Layer
        param.requires_grad = True
    for param in finetuned_model.layer4.parameters(): # Unfreeze Conv5_x
        param.requires_grad = True
    for param in finetuned_model.layer3.parameters(): # Unfreeze Conv4_x
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss() # Loss

    if train_flag: # Retrain the model
        
        finetuned_model = finetuned_model.to(device)
        
        optimizer = torch.optim.Adam([
            {"params": finetuned_model.fc.parameters(),     "lr": LR/10},  # head 1/10
            {"params": finetuned_model.layer4.parameters(), "lr": LR/10},  # late conv 1/10
            {"params": finetuned_model.layer3.parameters(), "lr": LR/10},  # late conv 1/10
        ], weight_decay=0.0001)
        # LR scheduler: reduce LR by a factor of 10 at plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
        
        # Loss & Acc History for Train, Val, Test Dataset
        tr_loss_history, tr_acc_history = [], []
        val_loss_history, val_acc_history = [], []
        te_loss_history, te_acc_history = [], []
        
        best_te_acc = 0.0
        best_te_loss = 0.0
        best_epoch = 0
        for epoch in range(EPOCH):
            tr_loss, tr_acc = Train(finetuned_model, tr_loader, criterion, optimizer)
            val_loss, val_acc = Test(finetuned_model, val_loader, criterion)
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
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
                
        torch.save(finetuned_model, save_model_path)
        plot_loss_history(tr_loss_history, val_loss_history, te_loss_history, tr_acc_history, val_acc_history, te_acc_history, str(Path(__file__).resolve().parent) + '/model_b.png', f"Model B(LR = {LR}, Batch Size = {BATCH_SIZE})")
        print(f"Best: Epoch {best_epoch + 1} / Test Acc: {round(best_te_acc * 100, 3)}% / Test Loss: {round(best_te_loss, 3)}")

    else: 
        try:
            finetuned_model = torch.load(save_model_path, map_location=device, weights_only=False) # Load the pretrained model
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
            print(f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")
        except FileNotFoundError:
            print(f"Pre-trained model doesn't exist at {save_model_path}")

def train_model_c(train_flag, save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE):
    save_model_path = save_path + "/model_c.pt"
        # Data Loadder
    tr_loader, val_loader, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE, data_aug=2)

    # Unfreeze All Parameters
    finetuned_model = copy.deepcopy(baseline)
    for param in finetuned_model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss() # Loss

    if train_flag: # Retrain the model
        
        finetuned_model = finetuned_model.to(device)
        
        optimizer = torch.optim.Adam([
            {"params": finetuned_model.fc.parameters(),     "lr": LR/10},  # head 1/10
            {"params": finetuned_model.layer4.parameters(), "lr": LR/10},  # late conv 1/10
            {"params": finetuned_model.layer3.parameters(), "lr": LR/10},  # late conv 1/10
            {"params": finetuned_model.layer2.parameters(), "lr": LR/100},  # early conv 1/100
            {"params": finetuned_model.layer1.parameters(), "lr": LR/100},  # early conv 1/100
            {"params": finetuned_model.conv1.parameters(), "lr": LR/100},  # stem conv 1/100
            {"params": finetuned_model.bn1.parameters(), "lr": LR/100},  # stem conv 1/100
        ], weight_decay=0.0001)

        # LR scheduler: reduce LR by a factor of 10 at plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
        
        # Loss & Acc History for Train, Val, Test Dataset
        tr_loss_history, tr_acc_history = [], []
        val_loss_history, val_acc_history = [], []
        te_loss_history, te_acc_history = [], []
        
        best_te_acc = 0.0
        best_te_loss = 0.0
        best_epoch = 0
        for epoch in range(EPOCH):
            tr_loss, tr_acc = Train(finetuned_model, tr_loader, criterion, optimizer)
            val_loss, val_acc = Test(finetuned_model, val_loader, criterion)
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
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
                
        torch.save(finetuned_model, save_model_path)
        plot_loss_history(tr_loss_history, val_loss_history, te_loss_history, tr_acc_history, val_acc_history, te_acc_history, str(Path(__file__).resolve().parent) + '/model_c.png', f"Model C(LR = {LR}, Batch Size = {BATCH_SIZE})")
        print(f"Best: Epoch {best_epoch + 1} / Test Acc: {round(best_te_acc * 100, 3)}% / Test Loss: {round(best_te_loss, 3)}")

    else: 
        try:
            finetuned_model = torch.load(save_model_path, map_location=device, weights_only=False) # Load the pretrained model
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
            print(f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")
        except FileNotFoundError:
            print(f"Pre-trained model doesn't exist at {save_model_path}")
        
def train_model_d(train_flag, save_path, model_d, train_path, test_path, LR, EPOCH, BATCH_SIZE):
    save_model_path = save_path + "/model_d.pt"
    # Data Loadder
    tr_loader, val_loader, te_loader, num_classes = load_face_dataset(train_path, test_path, batch_size=BATCH_SIZE, data_aug=0) # Only adding mlp layers -> weak data augmentation

    # Finetuning
    
    for param in model_d.parameters(): # Freeze All Layers
        param.requires_grad = False
    for param in model_d.fc1.parameters(): # Unfreeze FC1 Layer
        param.requires_grad = True
    for param in model_d.fc2.parameters(): # Unfreeze FC2 Layer
        param.requires_grad = True
    for param in model_d.fc3.parameters(): # Unfreeze FC3 Layer
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss() # Loss

    if train_flag: # Retrain the model
        
        model_d = model_d.to(device)
        
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_d.parameters()), lr=LR, weight_decay=0.0001)
        optimizer = torch.optim.Adam([
            {"params": model_d.fc3.parameters(), "lr": LR/10},  # pre_trained head 1/10
            {"params": model_d.fc2.parameters(), "lr": LR},  # new layer
            {"params": model_d.fc1.parameters(), "lr": LR},  # new layer 
        ], weight_decay=0.0001)
        
        # LR scheduler: reduce LR by a factor of 10 at plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
        
        # Loss & Acc History for Train, Val, Test Dataset
        tr_loss_history, tr_acc_history = [], []
        val_loss_history, val_acc_history = [], []
        te_loss_history, te_acc_history = [], []
        
        best_te_acc = 0.0
        best_te_loss = 0.0
        best_epoch = 0
        for epoch in range(EPOCH):
            tr_loss, tr_acc = Train(model_d, tr_loader, criterion, optimizer)
            val_loss, val_acc = Test(model_d, val_loader, criterion)
            te_loss, te_acc = Test(model_d, te_loader, criterion)
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
                
        torch.save(model_d, save_model_path)
        plot_loss_history(tr_loss_history, val_loss_history, te_loss_history, tr_acc_history, val_acc_history, te_acc_history, str(Path(__file__).resolve().parent) + '/model_d.png', f"Model D(LR = {LR}, Batch Size = {BATCH_SIZE})")
        print(f"Best: Epoch {best_epoch + 1} / Test Acc: {round(best_te_acc * 100, 3)}% / Test Loss: {round(best_te_loss, 3)}")

    else: 
        try:
            finetuned_model = torch.load(save_model_path, map_location=device, weights_only=False) # Load the pretrained model
            te_loss, te_acc = Test(finetuned_model, te_loader, criterion)
            print(f"Test Loss: {round(te_loss, 3)}, Test Acc: {round(te_acc*100, 3)}%")
        except FileNotFoundError:
            print(f"Pre-trained model doesn't exist at {save_model_path}")

# ============================== Model Load or Construction ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_path = str(Path(__file__).resolve().parent) + "/baseline.pt"
baseline = torch.load(pretrained_model_path, map_location=device, weights_only=False)

class ModelD(nn.Module):
    def __init__(self, baseline):
        super(ModelD, self).__init__()
        
        # Remove the last FC layer
        self.features = nn.Sequential(*list(baseline.children())[:-1])
        
        # Add 2 FC layers + final classifier with BN
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.fc3 = baseline.fc
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model_d = ModelD(copy.deepcopy(baseline)).to(device)

# ============================== Dataset Construction ==============================
# Hyperparameters from Baseline
BATCH_SIZE = 8
EPOCH = 30
LR = 5e-3
train_flag = [False, False, False, False] # If you want train the models, set the i-th element of train_flag to True

print(sum(p.numel() for p in baseline.parameters()))

print("="*20, "Model A", "="*20)
train_model_a(train_flag[0], save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE)
print("="*20, "Model B", "="*20)
train_model_b(train_flag[1], save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE)
print("="*20, "Model C", "="*20)
train_model_c(train_flag[2], save_path, baseline, train_path, test_path, LR, EPOCH, BATCH_SIZE)
print("="*20, "Model D", "="*20)
train_model_d(train_flag[3], save_path, model_d, train_path, test_path, LR, EPOCH, BATCH_SIZE)