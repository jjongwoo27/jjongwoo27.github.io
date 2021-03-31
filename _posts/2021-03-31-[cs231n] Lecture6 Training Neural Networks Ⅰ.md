---

date: 2021-03-31 08:30:00 +0900
categories : [Machine Learning]
tags: [Study]
type: note



---
<br/>

# Lecture6 : Training Neural Networks Ⅰ

# Activation Functions

## Sigmoid : 1/(1+e^(-x))

![activation_functions_sigmoid.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_functions_sigmoid.png)

- In negative, positive region of the sigmoid, it's essentially flat → gradient = 0(zero) → chain any upstream gradient coming down → multiply by near zero → **get very small gradient (gradient vanishing)**
- When the input to a neuron is always positive, the gradients on "w" is always all positive or negative ****→ **always move in the same direction (inefficient gradient updates)** → ****why we want **zero-mean data!**

![zig_zag_path.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/zig_zag_path.png)

## tanh

![activation_functions_tanhx.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_functions_tanhx.png)

- "tanh(x) looks very similar to the sigmoid, but it's squashing to the range [-1, 1] → **zero centered**
- still kill the gradient flow

## ReLU (Rectified Linear Unit) : max(0, x)

![activation_functions_ReLU.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_functions_ReLU.png)

- commonly used activation function
- ReLUs were starting to be used a lot around 2012 (AlexNet)
- positive half of inputs → saturation X  (active ReLU) / **negative → saturation O (dead ReLU; never update) → research problem but it's still doing okay for training networks**
- People like to initialize ReLU neurons with **slightly positive biases (e.g. 0.01) to solves the dying ReLU problem**

## Leaky ReLU

![activation_functions_Leaky_ReLU.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_functions_Leaky_ReLU.png)

- not flat in negative region → slight slope → very computationally efficient
- doesn't have dying ReLU problem

## ELU (Exponential Linear Units)

![activation_functions_ELU.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_functions_ELU.png)

## Maxout "Neuron"

![maxout.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/maxout.png)

- Does not have the basic form of dot product → nonlinearity
- Generalizes ReLU and Leaky ReLU
- Linear Regime, Does not saturate, Does not have dying problem
- problem : doubles the number of parameters and neuron

## In Practice

- **Use ReLU (Be careful with learning rates)**
- Try out Leaky ReLU, Maxout, ELU
- Try out tanh but don't expect much
- **Don't use sigmoid**

# Data Preprocessing

- zero-centering, normalizing...
- In the typical machine learning problems, normalize the data → all features are in the same range, contribute equally
- **In Image, do zero-centering the pixel data, not normalize** : generally for images right at each location, it already has relatively comparable scale and distribution (just want to apply convolutional networks spatially, have our spatial structure over the original image)
- In Image, it's not common to normalize variance, to do PCA or whitening

# Weight Initialization

- Initialization is important because the learning process may vary depending on **how you set the starting value of Weight**

![activation_statistics.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_statistics.png)

## First Idea: Small random numbers

- gaussian with zero mean and 1e-2 standard deviation

```python
W = 0.01*np.random.randn(D,H)
```

- works okay for small networks, but problems with deeper networks

### Example : Small Random Number

![activation_zero.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_zero.png)

### Example : 1

![activation_one.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/activation_one.png)

## Xavier Initialization

```python
W= np.random.ramdn(fan_in, fan_out) / np.sqrt(fan_in) # layer initalization
```

- good rule to initialize the weight
- the variance of the input to be the same as the variance of the output
- It is assumed that there's linear activations so when using the RELU nonlinearity it breaks → fan_in→fain_in/2

## He Initialization

```python
W= np.random.ramdn(fan_in, fan_out) / np.sqrt(fan_in/2) # layer initalization
```

![He_initialization.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/He_initialization.png)

# Batch Normalization

- consider a batch of activations at some layer
- compute the empirical mean and variance independently for each dimension → normalize
- **inserted after FC or Convolutional layers, and before nonlinearity**
- the distribution of inputs to the layer constantly readjust to new distributions

![batch_normalization.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/batch_normalization.png)

## Note

- improves gradient flow through the network
- allows higher learning rates
- reduces the strong dependence on initialization
- at test time BatchNorm functions differently : a single fixed empirical mean of activations during training is used

# Babysitting the Learning Process

## Step1: Preprocess the data

- Image → zero mean data

## Step2: Choose the architecture

- pick any architecture that we want to start with
- initialize the network → make sure that loss is reasonable
- sanity check(check the original loss) ex. softmax classifiers → loss = -log(1/c) (약 2.3)

## Step3: Train

- make sure that you can overfit very small portion of the training data
- start with small regularization and find **learning rate** (1e-3~1e-5) that makes the loss go down
    - loss not going down : learning rate too low
    - loss exploding : learning rate too high

# Hyperparameter Optimization

## Cross-validation Strategy

- Cross-validation: training on training set, and then evaluating on validation set
- pick values spread out apart → learn for only a few epochs → get hyperparameter which values are good or not , nothing happening? → adjust accordingly

## Random Search vs. Grid Search

- Grid layout : only able to some values, have missed where were the good regions
- Random Layout : get much more useful signal overall (have more samples of different values of the important variable)

![search.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/search.png)

## Monitor and visualize the loss curve

![loss_curve.png](/assets/pic/Lecture6 Training Neural Networks Ⅰ a842974250ff4ff2920b1522df90e9cb/loss_curve.png)

- The gap between training accuracy and validation accuracy is high? → might have **overfitting** → increase regularization strength
    - low? → increase model capacity
