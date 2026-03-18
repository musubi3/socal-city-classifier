# Southern California City Classifier 🌴🏙️

A PyTorch-based Convolutional Neural Network (CNN) designed to classify street-view images into six distinct Southern California cities. 

Built using a highly optimized, fine-tuned **MobileNetV3-Large** architecture, this model achieves **~93% validation accuracy** while remaining highly efficient and lightweight.

## 🎯 The Classification Task
The model is trained to identify structural, environmental, and infrastructure differences across the following six classes:
* Anaheim
* Bakersfield
* Los Angeles
* Riverside
* San Luis Obispo (SLO)
* San Diego

## 🧠 Model Architecture & Training
* **Base Model:** `MobileNetV3-Large` (Pre-trained on ImageNet1K)
* **Parameters:** ~5.4 Million
* **Optimization:** Adam Optimizer (Initial LR: 0.001) with a StepLR Scheduler (50% decay every 3 epochs).
* **Loss Function:** Cross-Entropy Loss
* **Data Augmentation:** To ensure robust feature extraction and prevent overfitting on lighting or weather conditions, the training pipeline utilizes random horizontal flips, 15-degree random rotations, and color jittering.

## 📂 Project Structure
```text
├── scripts/
│   ├── train.py          # Training pipeline with validation and plotting
│   └── predict.py        # Inference script for the autograder/test set
├── data/                 # Main training dataset (ignored in version control)
├── test_data/            # Unseen testing images for inference
├── models/               # Saved model weights
│   └── model_weights.pt  
├── requirements.txt      # Python dependencies
└── README.md
```

## ⚙️ Setup & Installation
1. Clone the repository and navigate to the project directory.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Training the Model
To train the model from scratch, ensure your images are located in the data/ directory and run:

```bash
python src/train.py
```

This will automatically split the data (80/20), train for 10 epochs and save the final weights to `models/model_weights.pt`.

### Running Inference
To run predictions on a folder of unseen images, ensure your trained weights are available and run:

```bash
python src/predict.py
```

This will load the custom weights into the MobileNetV3-Large shell, evaluate the images using batch processing (DataLoader), and print the predicted class for each file.