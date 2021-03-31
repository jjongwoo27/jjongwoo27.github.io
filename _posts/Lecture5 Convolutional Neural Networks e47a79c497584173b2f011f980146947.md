# Lecture5 : Convolutional Neural Networks

# Convolution Layer

## Fully Connected Layer

- 32*32*3 image pixel data → stretch all of the pixels out → multiply by a weight

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/fc_layer.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/fc_layer.png)

- 1 number: the result of taking a dot product between a row of W and the input (a 3072-dimensional dot product)
- can't preserve spatial structure of image

## Convolution Layer

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/cv_layer.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/cv_layer.png)

- Convolve the filter with the image (slide over the image spatially, computing dot products) → preserve spatial structure of image
- We can get different size outputs depending on how we choose to slide
- multiple filters →multiple activation maps
- layer가 깊어질수록 특수한 feature까지 뽑아낼 수 있음

# Convolution Neural Networks

## Stride, Padding

- Stride: a parameter of the neural network's filter that modifies the amount of movement
    - ex. stride=1 → the filter will move one unit at a time
- Output size = (n-f)/stride+1 (n=input size)
- Adjust padding and stride to be an integer in size. (because when applying convolutional layers, we tend to lose pixels on the image)
- Padding: add extra pixels of filler around the boundary of our input image → input에 담긴 정보를 잃지 않도록 해줌

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/padding.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/padding.png)

- **Output size = (n+2p-f)/stride+1** (p=pad, f=filter)

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/summary_conv_layer.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/summary_conv_layer.png)

## Pooling layer

- makes the representations smaller and more manageable
- operates over each activation map independently

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/pooling.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/pooling.png)

- **Max Pooling**: Calculate the maximum value for each patch of the feature map.

![Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/max_pooling.png](Lecture5%20Convolutional%20Neural%20Networks%20e47a79c497584173b2f011f980146947/max_pooling.png)

cf) Average pooling method smooths out the image and hence the sharp features may not be identified when this pooling method is used

- pooling보다는 stride를 활용하여 이미지의 크기를 줄이는 경우가 더 많아지고 있음
- global average connected layer