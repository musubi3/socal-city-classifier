import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import time
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


TRAIN_FOLDER = 'data' 

class SoCalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.classes = ['Anaheim', 'Bakersfield', 'Los_Angeles', 'Riverside', 'SLO', 'San_Diego']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        city = img_name.split('-')[0]
        label = self.classes.index(city)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data_full = SoCalDataset(data_dir=TRAIN_FOLDER, transform=train_transform)
    val_data_full = SoCalDataset(data_dir=TRAIN_FOLDER, transform=val_transform)

    indices = list(range(len(train_data_full)))
    random.shuffle(indices) 
    split_idx = int(0.8 * len(indices))

    train_dataset = Subset(train_data_full, indices[:split_idx])
    val_dataset = Subset(val_data_full, indices[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1') # <-- Changed here
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 6)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    epochs = 10
    losses = []

    print('Starting training...')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            losses.append(loss.item())
            
            if i % 50 == 49:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 50:.3f}')
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad(): 
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'>>> Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}% <<<')
        print('-' * 40)

        scheduler.step()

    end_time = time.time()
        
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    torch.save(model.state_dict(), './models/model_weights.pt')
    print(f'Training complete in {minutes} minutes and {seconds} seconds. Weights saved.')

    # with open('losses.json', 'w') as f:
    #     json.dump(losses, f)
    # print("Training data exported to losses.json")

    # plt.plot(losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Empirical Risk (Loss)')
    # plt.title('Training Curve')
    # plt.savefig('training_curve.png')
    # print('Training curve plot saved to training_curve.png')

if __name__ == '__main__':
    train_model()