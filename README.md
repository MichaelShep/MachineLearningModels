# Segmentation Recognition CNN

My Undergraduate Project for creating a joint CNN for both Face Segmentation and Recognition.
As part of this project, we will also create a CNN for each of the individual process so that these can be compared to the joint model so that we can evaluate its effectiveness

This project is created in Python using the library PyTorch.

### Segmentation Model

The segmentation model that we have created is based off the U-Net model which has been shown to have very good results on Segmentation tasks. We have trained our model using the CelebA-MaskHQ dataset which contains around 30,000 different face images and includes all the relevant output masks for all these images.