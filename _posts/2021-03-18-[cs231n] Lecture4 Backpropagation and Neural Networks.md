---

date: 2021-03-18 08:30:00 +0900
categories : [Machine Learning]
tags: [Study]
type: note



---
<br/>

# Lecture4 : Backpropagation and Neural Networks

# Backpropagation

- Computational Graph : a directed graph where the nodes correspond to mathematical operations

![computational_graph.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/computational_graph.png)

- Backpropagation : recursively use the chain rule in order to compute gradient with respect to every variable in the computational graph (useful in complex function)
- Get the local gradient of the node from the back (using **Chain Rule**)

![backpropagation.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/backpropagation.png)

## Example

![example.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/example.png)

![sigmoid_gate.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/sigmoid_gate.png)

- Able to group some of the nodes together into more complex nodes (ex. Sigmoid Gate)

## Patterns in backward flow

![patterns.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/patterns.png)

- ADD gate : gradient distributor
- MAX gate : gradient router (only flow through where the gradient come from)
- MIN gate : gradient switcher (lcal gradient is the value of the other variable)

# Vectorized Operation

- Gradients for vectorized code : Jacobian Matrix!
- size : (input size) * (output size) / form : Diagonal Matrix
- In practice we process an entire minibatch so Jacobian would technically be complex.

![A_vectorized_example.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/A_vectorized_example.png)

# Modularized Implementation : Forward, Backward API

- Forward: compute result of an operation and save any intermediates needed for gradient computation in memory
- Backward: apply the chain rule to compute the gradient of the loss function with respect to the inputs

```python
class MultiplyGate(object):
	def forward(x, y):
		z = x*y
		self.x = x # must keep these around!
		self.y = y
		return z
	def backward(dz)::
		dx = self.y * dz # [dz/dx * dL/dz]
		dy = self.x * dz # [dz/dy * dL/dz]
		return [dx, dy]
```

 

![multiply_gate.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/multiply_gate.png)

# Neural Networks

- Neural Networks stack layers in a hierarchical way to make up more complex non-linear function
- Add **Hidden Layer** : perform nonlinear transformations of the inputs entered into the network.
- Hidden Layer is located between the input and output of the algorithm, in which the function applies weights to the inputs and directs them through an activation function as the output.

![neural_network.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/neural_network.png)

![brain.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/brain.png)

- Example of activation functions

![activation_function.png](_posts/Lecture4 Backpropagation and Neural Networks 8aecbc5043b2451aa4c51d88201c6199/activation_function.png)
