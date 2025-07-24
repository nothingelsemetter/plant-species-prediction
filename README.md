# plant-species-prediction
This project was created to predict plant species using convolutional neural networks. In this paper, I illustrate how the model can be improved (or not) using machine learning techniques.
Data source:
https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species
Methods:
	CNN
Set up:
First model:
def MScnn(num_classes):
    # Feature extraction
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=5),    # Output: (128 - 5 + 1) = 124 → 6x124x124
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                # → 6x62x62
        nn.Conv2d(6, 16, kernel_size=5),   # → (62 - 5 + 1) = 58 → 16x58x58
        nn.ReLU(),
        nn.MaxPool2d(2, 2)                 # → 16x29x29
    )
    # Fully connected classifier
    classifier = nn.Sequential(
        nn.Linear(16 * 29 * 29, 120),      # Flattened features
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, num_classes)         # Output layer
    )s
    # Forward pass function
    def forward_function(x):
        x = feature_extractor(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = classifier(x)
        return x
Outcome:
Epoch [1/10], Loss: 3.3205, Train Acc: 12.29%, Test Acc: 20.07%
Epoch [2/10], Loss: 2.7619, Train Acc: 24.66%, Test Acc: 26.23%
Epoch [3/10], Loss: 2.3523, Train Acc: 34.03%, Test Acc: 29.54%
Epoch [4/10], Loss: 1.9357, Train Acc: 45.34%, Test Acc: 31.13%
Epoch [5/10], Loss: 1.5318, Train Acc: 55.89%, Test Acc: 30.93%
Epoch [6/10], Loss: 1.1150, Train Acc: 67.15%, Test Acc: 29.48%
Epoch [7/10], Loss: 0.7394, Train Acc: 78.32%, Test Acc: 29.54%
Epoch [8/10], Loss: 0.4189, Train Acc: 87.69%, Test Acc: 28.60%
Epoch [9/10], Loss: 0.2441, Train Acc: 92.76%, Test Acc: 28.70%
Epoch [10/10], Loss: 0.1371, Train Acc: 96.15%, Test Acc: 27.72%
Last model:
def MScnn(num_classes):
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=5),    # → 220x220
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                # → 110x110
        nn.Conv2d(6, 16, kernel_size=5),   # → 106x106
        nn.ReLU(),
        nn.MaxPool2d(2, 2)                 # → 53x53
    )
    classifier = nn.Sequential(
        nn.Linear(16 * 53 * 53, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, num_classes)
    )
    def forward_function(x):
        x = feature_extractor(x)
        x = torch.flatten(x, 1)
        x = classifier(x)
        return x
Outcome:
Epoch [1/10], Loss: 2.4640, Train Acc: 10.75%, Test Acc: 18.86%
Epoch [2/10], Loss: 2.2097, Train Acc: 23.09%, Test Acc: 25.85%
Epoch [3/10], Loss: 2.0342, Train Acc: 28.73%, Test Acc: 27.97%
Epoch [4/10], Loss: 1.9165, Train Acc: 33.02%, Test Acc: 29.87%
Epoch [5/10], Loss: 1.8056, Train Acc: 36.68%, Test Acc: 29.66%
Epoch [6/10], Loss: 1.6470, Train Acc: 42.03%, Test Acc: 32.42%
Epoch [7/10], Loss: 1.4319, Train Acc: 50.08%, Test Acc: 32.94%
Epoch [8/10], Loss: 1.0327, Train Acc: 64.22%, Test Acc: 32.63%
Epoch [9/10], Loss: 0.6522, Train Acc: 77.09%, Test Acc: 33.47%
Epoch [10/10], Loss: 0.3434, Train Acc: 88.08%, Test Acc: 33.47%
(No improvement)
Potential Improvements:
	Delete photos of non-existing plants
	Split existing plant species into variegated forms
	Increase the dataset size


