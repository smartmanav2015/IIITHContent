import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# Add more imports if required

# Sample Transformation function
# YOUR CODE HERE for changing the Transformation values.
#trnscm = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
trnscm = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(),  # Ensures input is single channel (if trained on grayscale)
    transforms.ToTensor()
])

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),       #Pads the input tensor using the reflection of the input boundary, it similar to the padding.
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)  # Final output embedding of size 5
        )

    # forward_once is for one image. This can be used while classifying the face images
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

##########################################################################################################
## Sample classification network (Specify if you are using a pytorch classifier during the training)    ##
## classifier = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear...)           ##
##########################################################################################################

# Definition of classes as dictionary
classes = ['person1', 'person2', 'person3', 'person4', 'person5', 'person6', 'person7']

# YOUR CODE HERE for pytorch classifier
classifier = nn.Sequential(
    nn.Linear(4096, 512),  # Input features from Siamese network (4096) to 512
    nn.BatchNorm1d(512),    # Batch normalization for better training
    nn.ReLU(),            # ReLU activation function
    nn.Linear(512, 128),   # Further reduce to 128 features
    nn.ReLU(),            # ReLU activation
    nn.Linear(128, len(classes))  # Output layer with number of classes
)

