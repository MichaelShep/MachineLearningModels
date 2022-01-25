# Machine Learning Models

My Undergraduate Project for creating a joint CNN for both Face Segmentation and Attribute detection.
As part of this project, we will also create a CNN for each of the individual process so that these can be compared to the joint model so that we can evaluate its effectiveness

This project is created in Python using the library PyTorch.

The aim of this project is to compare the effectiveness of a multi-learning framework when compared with standard single use frameworks

### Dataset

For all the networks in this project we have used the same dataset, CelebAMask-HQ. This dataset contains 30,000 different face images, all of which come with segmentation and attribute data. We use 20,000 of these images for training our models and the other 10,000 for performing validation.

### Segmentation Model

The segmentation model that we have created is based off the U-Net model which has been shown to have very good results on Segmentation tasks. 

### Attributes Model

The attributes model we have created is loosely based of AlexNet whilst also adding elements from ResNet.

### Multi-Task Model

The multi-learning model uses concepts shown in the HyperFace network in order to be able to combine the two previous networks together in order to create a network that can be trained for both tasks at the same time.