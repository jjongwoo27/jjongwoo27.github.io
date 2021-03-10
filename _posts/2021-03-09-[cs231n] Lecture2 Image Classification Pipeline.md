---

date: 2021-03-09 08:30:00 +0900
categories : [Machine Learning]
tags: [Study]
type: note



---
### 2021 YAI Spring Session Week 1 : https://youtu.be/OoUX-nOEjG0


<br/>
 
# 1. Image Classification : A core task in Computer Vision

- An image in computer : just a big grid of numbers between [0, 225] (3D arrays, RGB)

## Challenges

- Semantic Gap  : the difference between semantic labels and pixel data
- Viewpoint Variation : all pixels change when the camera moves (rotate, zoom in/out)
- Illumination : different light condition
- Deformation : same object but various positions
- Occlusion : only see part of objects
- Background Clutter : foreground could be actually quite similar in appearance to the object
- Intraclass Variation : objects come in different shape, sizes and colors etc..<br/>

<br/>
 
# 2. Image Classifier

- There is no obvious way to hard-core algorithm for recognizing objects

```python
def classify_image(image):
	# Some magic here? NO!
	return class_label
```

## Data-Driven Approach

1. **Collect** datasets of images and labels
2. Use **Machine Learning** to train a classifier
3. **Evaluate** the classifier on new images

```python
def train(images, labels):
	# Machine Learning!
	return model

def predict(model, test_images):
	# Use model to predict labels
	return test_labels
```

<br/>
 
# 3. Nearest Neighbor

- Memorize all data and labels
- Predict the label of the most similar training image

## Distance Metric to compare images

- Use **L1 distance (Manhattan)** : input feature in vector have some important meaning

![L1_distance.png](/assets/pic/Lecture2 Image Classification Pipeline/L1_distance.png)

- **L2 distance (Euclidean)** : generic, don't know actually mean


## Implementation in Python

```python
import numpy as np

class NearestNeighbor
	def __init__ (self):
		pass
	
	""" Memorize training data """
	def train(self, X, y)
	""" X is N * D where each row is an example. Y is 1-dimension of size N """
	# the neares neighbor classifier siply remebers all training data
	self.Xtr=X
	self.Ytr=y

	""" For each test image: Find closest train image, Predict label of nearest images """
	def predict (self, X):
	""" X is N * D where each row is an example we wish to predict label for """
	num_test=X.shape[0]
	# lets make sure that the output type matches the input type
	Ypred=np.zeros(num_test, dtype=self.ytr.dtype)
	
		# loop over all test rows
		for i in xrange(num_test):
			# find the nearest training image to the i'th test images
			# using the L1 distance (sum of abolute value differences)
			distance=np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
			min_index=np.argmin(distances) # get the index with smallest distance
			Ypred[i]=self.ytr[min_index] # predict the label of the nearest example

	return Ypred
```

- N examples → Train : O(1), Predict O(N) → fast at training, slow for prediction
- **We want classifiers that are fast at prediction; slow for training**
- **Nearest Neighbor → not suitable for image classifier**

<br/>
 
# 4. K-Nearest Neighbors

- Instead of copying label from nearest neighbor, take majority vote from K closest points (majority vote)
- **Hyperparameter** : value of K,  Distance Metric(, Voting), **parameters selected by the algorithm rather than learning from training data** → try them all and see what works better

![k_nearest.png](/assets/pic/Lecture2 Image Classification Pipeline/k_nearest.png)

- L1(Manhattan) distance : follow coordinate axis, depends on your choice of coordinate system
- L2(Euclidean) distance : not care about coordinate axis (naturally)

## k-Nearest Neighbors on images never used.

- Very slow at test time
- Distance metrics on pixels are not informative for measuring the perceptual similarity
- Curse of dimensionality (exponential growth in computation when expanding dimension)

<br/>
 
# 5. Setting Hyperparameters

![setting_hyperparameter_1.png](/assets/pic/Lecture2 Image Classification Pipeline/setting_hyperparameter_1.png)

- Idea #1, #2 : poor performance on new data that was not in training data
- Idea #3 : Validation set is used to measure the accuracy without knowing label → can be checked from data that has never been seen before
- **Idea #3 is best method for high accuracy**

![setting_hyperparameter_2.png](/assets/pic/Lecture2 Image Classification Pipeline/setting_hyperparameter_2.png)

- Idea #4 : likely to find optimal parameter(time-consuming method)  → used for small data

<br/>
 
# 6. Linear Classification

- Linear Classification : basic building block of neural network
- Parametric Approach

![Linear.png](/assets/pic/Lecture2 Image Classification Pipeline//Linear.png)

- 3072 : the numbers of pixels, 10 : the numbers of classes
- image → stretch pixels into column → inner product with weight, add bias → show similarity between templates for the class and the pixels of images (template matching)

## Problem

- Only learning one template for each class → low accuracy of complex models like neural networks
- Hard classes for a linear classifier → Linear classifier makes a line that separate each class from the rest and views the image as a point in high-dimensional space

![hard_case_linear.png](/assets/pic/Lecture2 Image Classification Pipeline/hard_case_linear.png)
