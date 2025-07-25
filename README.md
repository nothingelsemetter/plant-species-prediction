# :herb: Plant species classification
This project was created to predict plant species using convolutional neural networks. In this work, I illustrate how the model can be improved (or not) using machine learning techniques.

Dataset

Sourse: https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species

The dataset consists of 47 plant species.

Processing steps include:

* Deleting non-existing or corrupted images
* Image augmentation 
* Normalization

Model Architecture

The classification model is based on a Convolutional Neural Network (CNN), built and trained using PyTorch.

Key components include:
* Multiple convolutional layers with ReLU activation
* MaxPooling 
* Dropout 
* Fully connected output layer with softmax activation

For reference, I used pre-trained models VGG16 for accuracy improvement.

Training

* Loss Function: CrossEntropyLoss
* Optimizer: Adam / SGD

Potential steps for improvement

* Splitting species into variegated and non-variegated forms
* Delete artificially created images
* Increase number of images for each species



