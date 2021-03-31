# Lecture7 : Training Neural Networks Ⅱ

# Optimization: Problems with SGD

- **Loss Function이 steep vertically but shallow horizontally하다고 가정하자. 이때, SGD는 어떻게 진행될까?**

    ![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/zigzag.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/zigzag.png)

    - shallow한 차원에서는 천천히, steep한 차원에서는 가파르게 진행된다. (characteristic zigzagging behavior) → 원하는 대로 학습이 진행되지 않을 수도 있다.
    - 고차원의 학습, 많은 parameters를 학습시킬 경우에 차원 별로 maximum과 minimum이 큰 차이가 발생할 수 있어 큰 문제가 될 수 있다.

- **Loss Function이 local minima 또는 saddle point를 갖는다고 가정하자. 이때, SGD는 어떻게 진행될까?**
    - local minima, saddle point에서 gradient는 0이 되기 때문에 gradient가 update되지 않을 수 있다. (get stuck)
    - saddle point 주변은 gradient가 0은 아니지만 기울기가 매우 작기 때문에 학습이 매우 천천히 진행될 수 있다.
    - 고차원의 학습일수록 saddle point가 더 발생할 가능성이 높아 문제가 될 수 있다.

  

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/saddle_point.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/saddle_point.png)

# SGD + Momentum

- SGD에 Momentum 개념을 더해서 위와 같은 문제를 해결할 수 있다.

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/momentum.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/momentum.png)

- SGD에서는 gradient 자체를 직접 업데이트 하지만, SGD+Momentum은 "v(velocity)"와 "rho(friction)"을 도입하여 이를 통해 업데이트한다.
- current velocity → decay it by friction → add in our gradient (학습 방향도 velocity vector 방향으로)
- 따라서 local minima, saddle point에서도 velocity를 가지므로 gradient를 계속 update시켜나갈 수 있다. → SGD에 비해 smooth한 학습이 진행된다.

## Nesterov Momentum

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/nesterov_1.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/nesterov_1.png)

- Nesterov는 기존 Momentum과 다르게 Gradient Step을 원래 지점에서 취하지 않고 Momentum Step이 이동한 걸로 예상되는 지점(벡터의 종점)에서 취한다.
- Convex Optimization에서 매우 좋은 성능을 보인다. (Neural Network와 같은 Non-Convex Problem에서는 미미한 편)

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/nesterov_2.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/nesterov_2.png)

- Nesterov는 현재 위치가 아닌 다른 위치에서의 gradient를 계산해야하므로 기존 식을 변형한 형태로 사용한다.

 

# AdaGrad

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/adagrad.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/adagrad.png)

- 파라미터 마다 학습된 횟수가 다르기 때문에 각각의 학습 횟수를 기억해 서로 다른 running estimate을 반영한다.
- Convex한 경우 좋은 성능을 보이지만 Non-convex한 경우 saddle point같은 곳에서 더 이상 학습이 진행되지 않는 문제가 발생할 수 있다.

## RMSProp

- AdaGrad의 문제점을 해결한 것이 RMSProp
- Momentum을 추가한 것과 같은 효과가 나타난다

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled.png)

# Adam

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Adam.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Adam.png)

- RMSProp과 Momentum 개념을 합한 방식으로 가장 많이 쓰이는 방식 중 하나이다.
- 분모에 작은 양의 상수를 추가하여 초기 moment가 0으로 초기화될 때 0으로 나누는 일이 발생하지 않도록 조정해준다. (Bias Correction)

# Learning Rate

- SGD, SGD+Momentum, Adagrad, RMSProp, Adam 모두는 **learning rate**를 hyperparameter로 가진다.

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%201.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%201.png)

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/deacy_graph.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/deacy_graph.png)

- 초기에는 높은 learning rate로 학습을 시켜나가다가, 시간에 따라 learning rate를 decay해가면서 섬세하게 조절해 나가면 학습에 도움이 된다.
- learning rate는 second-order hyperparameter이므로 처음부터 decay를 진행하면 안된다. 학습을 일단 진행한 다음에 필요한 부분에서 decay하는 것이 좋다.

# Second-Order Optimization

- First-Order Optimization: linear approximation으로부터 gradient를 구한다. → very large region의 경우 적용하기 어려움 (only incorporating information about the first derivative of the function)
- Second-Order Optimization: gradient와 Hessian행렬을 이용하여 quadratic approximation을 형성한다.

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%202.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%202.png)

- 계속 minimum을 찾아가면 되므로 learning rate를 설정해줄 필요가 없다. (하지만 실제 학습에서는 quadratic approximation이 완벽하지 않기 때문에 필요하긴 하다.)
- Deep Learning에서는 parameter가 무수히 많으므로 이를 Hessian행렬로 나타내기에는 너무 크기 때문에 자주 사용하지 않는다. → Quasi-Newton methods(BGFS)를 대신 사용
- L-BFGS (Limited memory BFGS): full batch, deterministic mode에서 주로 사용, 딥러닝과 같이 mini-batch setting에서는 좋은 결과가 나오지 못한다.

cf. In practice: **Adam** is good default choice in most cases. If you can afford to do full batch updates then try out L-BFGS (and don't forget to disable all sources of noise)

# Reducing Training Error

- What can we do to try to reduce gap between train and validation accuracy and make our model perform better on unseen data?

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/training_error.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/training_error.png)

## Model Ensembles

- 여러 가지 모델을 각각 학습하고 정확도를 평균을 낸다 → 엄청난 학습 향상보다는 점진적으로 성능이 올라간다. (실제로 흔하게 사용하는 방법)
- Tips and Tricks: Instead of training independent models, use multiple snapshots of a single model during training

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/model_ensembles.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/model_ensembles.png)

- Polyak average: a smooth ensemble of your own network during training → smoothly decaying average of parameter vector (not too common in practice)

## Regularization

- training data에 overfitting되는 것을 방지하고 성능을 높이기 위해 추가하는 term
- 이전 chapter에서 배웠던 L1, L2 Regularization도 있지만 Neural Net에는 잘 어울리지 않음

### Dropout

- 매 train마다, 각 layer에서 임의로 일부 neurons를 0으로 만든다. test 과정에서 모든 뉴런을 사용한다. (average out the randomness, multiply output by the dropout probability)

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%203.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/Untitled%203.png)

- 특정 뉴런의 가중치에 영향을 덜 받아 overfitting이 감소할 수 있다.
- Dropout은 parameter를 공유하는 거대한 ensemble을 학습시키는 것과 같은 효과를 나타낸다고 볼 수 있다.

### Data Augmentation

- train 과정에서 데이터를 무작위로 변형(이미지 자르기, 뒤집기, 회전하기, 명암 조절 등)하여 학습하여 같은 레이블의 더 많은 데이터를 학습시킬 수 있다.

![Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/data_augmentation.png](Lecture7%20Training%20Neural%20Networks%20%E2%85%A1%20e5e79495358e4e6796e67467fd8c9266/data_augmentation.png)

### Transfer Learning

- CNN 기반 딥 러닝 모델을 제대로 훈련시키려면 많은 수의 데이터가 필요하지만 이를 얻는 것은 쉽지 않다.
- 이를 해결하기 위해 이미 훈련된 알고리즘 모델을 가지고 와서 내가 하려고 하는 학습에 맞게 layer를 일부 수정하여 적은 이미지 데이터로도 학습시키는 방법.
- 이떄, 가중치를 바꾸지 않는 layer에는 freeze 해주어야 한다.