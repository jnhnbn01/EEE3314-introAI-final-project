"""
Question 4: Complete Implementation with Fixed Indexing
FCResNet18 for training-free object detection
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------- Fully Convolutional ResNet18 (FCResNet18) --------------------

class FCResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(FCResNet18, self).__init__()
        
        resnet18 = models.resnet18(pretrained=True) # Load pretrained ResNet18

        # Copy feature extractor
        self.conv1 = resnet18.conv1 
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        # Problem 1-(a) Apply average pooling instead of GAP
        # Using avgpool2d -> nxmx512 instead of 1x1x512; (H - 7 + 2*3)/1 + 1 = H
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3) 
        
        # Problem 1-(b) Change output layer to produce nxmx1000
        # Convert FC layer to 1x1 convolution
        self.fc_conv = nn.Conv2d(512, num_classes, kernel_size=1)
         
        # Initialize with pretrained weights for last 
        fc_weights = resnet18.fc.weight.data
        fc_bias = resnet18.fc.bias.data
        self.fc_conv.weight.data = fc_weights.view(num_classes, 512, 1, 1)
        self.fc_conv.bias.data = fc_bias
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, n=H/32, m=W/32]
        x = self.avgpool(x)  # [B, 512, n, m]
        x = self.fc_conv(x)  # [B, 1000, n, m]
        return x

# -------------------- Make class dictionary of Imagenet Dataset --------------------

def load_imagenet_classes(file_path=str(Path(__file__).resolve().parent) + '/imagenet_classes.txt'):
    classes = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            comma_idx = line.find(',') # find ,
            if comma_idx != -1 and line[:comma_idx].isdigit(): 
                idx = int(line[:comma_idx]) # split into two parts by comma
                class_name = line[comma_idx + 1:].strip()
                classes[idx] = class_name

    return classes

# -------------------- Inference --------------------

def perform_inference(model, image_path):
    print(f"Image Path: {image_path}")
    
    
    image = Image.open(image_path).convert('RGB') # Load image
    original_image = cv2.imread(image_path) 
    original_image_shape = original_image.shape
    labels = load_imagenet_classes() # ImageNet Class Dictionary
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]) # Nomalization with mean, std of ImageNet Dataset
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(image_tensor) 
        preds = torch.softmax(preds, dim=1) # get a probability map
        pred, class_idx = torch.max(preds, dim=1) # get a class of each patch
        
        print(f"Response map shape: {preds.shape}")
        print(f"class_idx shape: {class_idx.shape}")

        # Find position with maximum activation across all classes
        max_activation, _ = torch.max(preds.view(1, 1000, -1), dim=2)  # [1, 1000]
        predicted_class = torch.argmax(max_activation).item()

        print(f"Predicted class: {predicted_class}")
        
        # where is the maximum for this class
        class_response = preds[0, predicted_class, :, :]
        max_val = torch.max(class_response)
        max_pos = torch.where(class_response == max_val)
        if len(max_pos[0]) > 0:
            print(f"Max response at position: ({max_pos[0][0].item()}, {max_pos[1][0].item()})")
        
        
        print(f"Predicted Class: {predicted_class} - {labels[predicted_class]}") # Print top predicted class
    
    return preds, predicted_class, original_image, labels

# -------------------- Visualization --------------------

def visualization(preds, predicted_class, original_image, labels, print_image, image_path, threshold=0.25):
    """
    Step 3: Overlay response map following the provided code snippet exactly
    """
    print(f"\n=== Step 3: Overlay Response Map ===")
    
    # Following the provided code snippet exactly:
    
    # Find the n x m score map for the predicted class
    score_map = preds[0, predicted_class, :, :].cpu().numpy()
    score_map_original = score_map.copy()
    print(f"Original score map shape: {score_map_original.shape}")
    
    # Resize score map to the original image size
    score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))
    print(f"Resized score map shape: {score_map.shape}")
    
    # Binarize score map
    _, score_map_for_contours = cv2.threshold(score_map, threshold, 1, cv2.THRESH_BINARY)
    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()
    
    # Find the contour of the binary blob
    contours, _ = cv2.findContours(score_map_for_contours, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Find bounding box around the object
    rect = None
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(largest_contour)
        print(f"Bounding box: {rect}")
    else:
        print("No contours found")
    
    # Apply score map as a mask to original image
    score_map_norm = score_map - np.min(score_map)
    if np.max(score_map_norm) > 0: # prevent dividing with zero
        score_map_norm = score_map_norm / np.max(score_map_norm)
    
    # Following the display code snippet:
    score_map_color = cv2.cvtColor((score_map_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    masked_image = (original_image * score_map_norm[:, :, np.newaxis]).astype(np.uint8)
    
    # Display bounding box
    display_image = masked_image.copy()
    if rect:
        x, y, w, h = rect
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Scaled score map
    im = axes[0, 1].imshow(score_map, cmap='viridis')
    axes[0, 1].set_title(f"Scaled Score Map\nClass {predicted_class}: {labels[predicted_class]}")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad = 0.03, shrink = 0.75 * score_map.shape[0] / score_map.shape[1])
    
    # Activations and bbox
    axes[0, 2].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Activations and BBox")
    axes[0, 2].axis('off')
    
    # Response map (original size from model)
    axes[1, 0].imshow(score_map_original, cmap='gray')
    axes[1, 0].set_title(f"Response Map (Model Output)\nSize: {score_map_original.shape}")
    axes[1, 0].axis('off')
    
    # Resized response map (as shown in the problem)
    axes[1, 1].imshow(score_map, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Resized Response Map\nSize: {score_map.shape}")
    axes[1, 1].axis('off')
    
    # Detected object (like the example output)
    detected_object = np.zeros_like(original_image)
    if rect:
        x, y, w, h = rect
        detected_object[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]
        cv2.rectangle(detected_object, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    axes[1, 2].imshow(cv2.cvtColor(detected_object, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Detected Object")
    axes[1, 2].axis('off')
    
    plt.suptitle("FCResNet18 Object Detection Results", fontsize=16)
    plt.tight_layout()
    plt.savefig(image_path[:-4] + '_result.png', dpi=100, bbox_inches='tight')
    if print_image:
        plt.show()
    
    return score_map_original, rect

# -------------------- Whole Detection & Classification Pipeline --------------------

def test_image(model, image_path, labels, print_image):
    """Test a single image and return results"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    try:
        # Perform inference
        preds, predicted_class, original_image, _ = perform_inference(model, image_path)
        
        # Visualize
        score_map, bbox = visualization(preds, predicted_class, original_image, labels, print_image, image_path)
        
        # Get image size
        img = Image.open(image_path)
        img_size = img.size
        
        return {
            'class_id': predicted_class,
            'class_name': labels[predicted_class],
            'response_map_size': f"{score_map.shape[0]}x{score_map.shape[1]}",
            'input_size': f"{img_size[0]}x{img_size[1]}",
            'bbox': bbox
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# -------------------- Main --------------------

def main():
    print("="*70)
    print("Question 4: Fully Convolutional ResNet18 Object Detector")
    print("="*70)
    
    # Load ImageNet classes
    labels = load_imagenet_classes()
    print_image = True
    
    # Initialize FCResNet18
    print("\n" + "="*70 + "\nQuestion 4-1: Initialize FCResNet18\n" + "="*70 )
    model = FCResNet18().to(device)
    model.eval()
    print("✓ FCResNet18 loaded successfully")
    
    # Test with camel image
    print("\n" + "="*70 + "\nQuestion 4-2 ~ 4-3: Testing with Arabian Camel Image\n" + "="*70 )
    camel_path = str(Path(__file__).resolve().parent) + '/images/camel-1536x580.jpg'
    camel_result = test_image(model, camel_path, labels, print_image)

    # Student ID based detection
    print("\n" + "="*70 + "\nQuestion 4-4: Student ID Based Detection\n" + "="*70)
    
    student_id = "2020132002"  # Replace with actual student ID
    Z = int(student_id[-3:])
    print(f"Student ID last 3 digits: Z = {Z}")
    print(f"Class Z = {Z}:{labels[Z]}")
    print(f"Class Z+5 = {Z+5}: {labels[Z+5]}")
    
    # Z image
    z_path = str(Path(__file__).resolve().parent) + '/images/z.jpg' # https://www.oceans-research.com/all-about-sharks/
    z_result = test_image(model, z_path, labels, print_image)
    
    # Z+5 image
    z5_path = str(Path(__file__).resolve().parent) + '/images/zplus5.jpg' # https://m.ruliweb.com/community/board/300143/read/71655984
    z5_result = test_image(model, z5_path, labels, print_image)
    
    # Face and random images
    print("\n" + "="*70 + "\nQuestion 4-5: Test 1 Face and 2 Random Images\n" + "="*70)
    
    # Face Image
    face1_path = str(Path(__file__).resolve().parent) + '/images/face_chaeunwoo.png'
    face1_result = test_image(model, face1_path, labels, print_image)
    
    # Ballplayer Image
    face2_path = str(Path(__file__).resolve().parent) + '/images/face_ryuhyunjin.png'
    face2_result = test_image(model, face2_path, labels, print_image)
    
    # Random Image 1
    random1_path = str(Path(__file__).resolve().parent) + '/images/random1.png'
    random1_result = test_image(model, random1_path, labels, print_image)   
    
    random1_crop1_path = str(Path(__file__).resolve().parent) + '/images/random1_crop1.png'
    random1_crop1_result = test_image(model, random1_crop1_path, labels, print_image)   
    
    random1_crop2_path = str(Path(__file__).resolve().parent) + '/images/random1_crop2.png'
    random1_crop2_result = test_image(model, random1_crop2_path, labels, print_image)   
    
    random1_crop3_path = str(Path(__file__).resolve().parent) + '/images/random1_crop3.png'
    random1_crop3_result = test_image(model, random1_crop3_path, labels, print_image)   
    
    # Random Image 2
    random2_path = str(Path(__file__).resolve().parent) + '/images/random2.jpg'
    random2_result = test_image(model, random2_path, labels, print_image)
    
    # Results table
    print("\n" + "="*70 + "\nRESULTS TABLE\n" + "="*70)
    print(f"{'Student ID':<12} | {'Predicted Class Label':<20} | {'Response Map Size':<18} | {'Display Input Image':<25} | {'Detection':<10}")
    print("-"*100)
    
    if camel_result:
        print(f"{camel_result['class_id']:<12} | {labels[camel_result['class_id']]:<20} | {camel_result['response_map_size']:<18} | "
              f"{'Image ('+camel_result['input_size']+')':<25} | {'Yes' if camel_result['bbox'] else 'No':<10}")
    if z_result:
        print(f"{z_result['class_id']:<12} | {labels[z_result['class_id']]:<20} | {z_result['response_map_size']:<18} | "
              f"{'Image ('+z_result['input_size']+')':<25} | {'Yes' if z_result['bbox'] else 'No':<10}")
    if z5_result:
        print(f"{z5_result['class_id']:<12} | {labels[z5_result['class_id']]:<20} | {z5_result['response_map_size']:<18} | "
              f"{'Image ('+z5_result['input_size']+')':<25} | {'Yes' if z5_result['bbox'] else 'No':<10}")
    if face1_result:
        print(f"{'-':<12} | {'Face image 1':<20} | {face1_result['response_map_size']:<18} | {'Image ('+face1_result['input_size']+')':<25} | {'Yes' if face1_result['bbox'] else 'No':<10}")
    if face2_result:
        print(f"{'-':<12} | {'Face image 2':<20} | {face2_result['response_map_size']:<18} | {'Image ('+face2_result['input_size']+')':<25} | {'Yes' if face2_result['bbox'] else 'No':<10}")
    if random1_result:
        print(f"{'-':<12} | {'Random image 1':<20} | {random1_result['response_map_size']:<18} | {'Image ('+random1_result['input_size']+')':<25} | {'Yes' if random1_result['bbox'] else 'No':<10}")
    if random1_crop1_result:
        print(f"{'-':<12} | {'Random image 1 Cropped 1':<20} | {random1_crop1_result['response_map_size']:<18} | {'Image ('+random1_crop1_result['input_size']+')':<25} | {'Yes' if random1_crop1_result['bbox'] else 'No':<10}")
    if random1_crop2_result:
        print(f"{'-':<12} | {'Random image 1 Cropped 2':<20} | {random1_crop2_result['response_map_size']:<18} | {'Image ('+random1_crop2_result['input_size']+')':<25} | {'Yes' if random1_crop2_result['bbox'] else 'No':<10}")
    if random1_crop3_result:
        print(f"{'-':<12} | {'Random image 1 Cropped 3':<20} | {random1_crop3_result['response_map_size']:<18} | {'Image ('+random1_crop3_result['input_size']+')':<25} | {'Yes' if random1_crop3_result['bbox'] else 'No':<10}")
    if random2_result:
        print(f"{'-':<12} | {'Random image 2':<20} | {random2_result['response_map_size']:<18} | {'Image ('+random2_result['input_size']+')':<25} | {'Yes' if random2_result['bbox'] else 'No':<10}")
        
if __name__ == "__main__":
    main()