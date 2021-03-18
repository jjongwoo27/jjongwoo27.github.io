---

date: 2021-03-18 08:30:00 +0900
categories : [Machine Learning]
tags: [Study]
type: note



---
<br/>
 

# Multiclass SVM Loss

- Loss function tells how good our current classifiers is.

![loss_function.png](/assets/pic/Lecture3 Loss Functions and Optimization/loss_function.png)

- Multiclass SVM Loss : sum over all of categories Y, except for the true categories Y_i
- s_j: the score of predicted class , s_yi: the score of the true class
- Safety margin(=1) is set to give a difference between the predicted value and the correct value. → Correct class score must be greater than the other scores than a certain safety margin
- Max : ∞, min : 0 (all correct)

![svm_loss.png](/assets/pic/Lecture3 Loss Functions and Optimization/svm_loss.png)

- At an initialization W is small so all s→0, the loss is "(the number of the classes) - 1"
- In Multiclass SVM loss, the true value doesn't mean anything. It is important that getting the correct class scores to be greater than  one more the incorrect score.

<br/>
 
# Regularization

- In any Loss Function, "W" is not unique. → choose value that fits well in test data
- Add **penalty(Regularization)** to the loss function to encourages the model to pick a simpler W and prevents the model from overfitting in train data
- $\lambda$ : regularization strength (hyperparameter)

![regularization.png](/assets/pic/Lecture3 Loss Functions and Optimization/regularization.png)

## Commonly used Regularization Form

- L2 Regularization ★ : consider all data, weights
- L1 Regularization : consider only important weights → feature selection O, sparse model, convex optimization
- Elastic Net (L1+L2)
- Max Norm Regularization, Dropout, Batch Normalization, Stochastic Depth...

![common_used_regularization.png](/assets/pic/Lecture3 Loss Functions and Optimization/common_used_regularization.png)

<br/>
 
# Softmax Classifier (Multinomial Logistic Regression)

- Endow scores with some additional meaning
- Scores = unnormalized log probabilities of the classes
- Make probability of the true class high and incorrect class low
- Max : ∞, min : 0 (all correct)
- At an initialization W is small so all s→0, the loss is "logC"

![softmax.png]/assets/pic/Lecture3 Loss Functions and Optimization/softmax.png)

## Softmax vs. SVM

- Q: Suppose I take a datapoint and I jiggle a bit (changing its score slightly). What happens to
the loss in both cases?
- SVM : not change
- Softmax : pile more probability mass on the correct class (sensitive)

<br/>
 
# Optimization

- Ways to find the best "W" (loss → 0)
- Strategy #1 Random Search : possible but inefficient
- Strategy #2 Gradient Descent : down the slope to find the best W

## Gradient Descent

- gradient : the vector of (partial derivatives) along each dimension
- Compute gradient to determine which direction go down by moving along the slope. (the direction of steepest descent is the negative gradient)
- Numerical Gradient : approximate, slow, easy to write vs. Analytic Gradient : exact, fast, error-prone → **Always use Analytic Gradient but check implementation with Numerical Gradient (Gradient Check!!!)**

```python
# Vanila Gradient Descent

while True:
	weights_grad = evaluate_gradient(loss_fun, data, weights)
	weights +- = - step_size * weights_grad # Perform parameter update
```

## Stochastic Gradient Descent (SGD)

- Computing loss could be actually very expensive and slow
- Update W using a minibatch of data rather than all data in every iteration

```python
# Vanila Minibatch Gradient Descent

while True:
	data_batch = sampling_training_data(data, 256) # sample 256 examples
	weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
	weights +- = - step_size * weights_grad # Perform parameter update
```

<br/>
 
# Image Features

- Extract features (form of vector) from raw image pixels → **Classifier Model or Neural Network**
- Color Histogram : which color was used the most
- Histogram of Oriented Gradients(HoG) : Which type of edge was used the most
- Bag of Words : Cut images into multiple patches, collect images in the form of a codebook, and create a histogram based on the corresponding values for incoming images.
