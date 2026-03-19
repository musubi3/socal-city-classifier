import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

CLASSES = sorted(['Anaheim', 'Bakersfield', 'Los_Angeles', 'Riverside', 'SLO', 'San_Diego'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

class InferenceDataset(Dataset):
    '''Custom Dataset to load test images for batch processing.'''
    def __init__(self, test_dir):
        self.test_dir = pathlib.Path(test_dir)
        self.image_paths = sorted(list(self.test_dir.glob('*.jpg')))
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        return tensor, path.name

def predict(test_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(CLASSES))

    weights_path = './models/model_weights.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = InferenceDataset(test_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions = {}
    
    with torch.no_grad():
        for tensors, filenames in dataloader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            predicted_indices = outputs.argmax(dim=1)
            
            for i in range(len(filenames)):
                predictions[filenames[i]] = CLASSES[predicted_indices[i].item()]

    return predictions

if __name__ == '__main__':
    preds = predict('./data')
    print('Predictions:')
    
    correct = 0
    total = len(preds)
    
    for filename, predicted_label in sorted(preds.items()):
        true_label = filename.split('-')[0]
        
        if predicted_label == true_label:
            correct += 1
            print(f'✅ {filename}: {predicted_label}')
        else:
            print(f'❌ {filename}: Predicted {predicted_label} (Actual: {true_label})')
            
    print('-' * 40)
    print(f'Test Accuracy: {(correct / total) * 100:.2f}% ({correct}/{total})')