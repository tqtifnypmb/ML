### Notes

Dataset [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/car+evaluation)

---

#### Multilayer perceptron model

MLP 可以理解成一系列 perceptron model 的组合，Hidden layer 中的每一个 unit 是一个 perceptron model。MLP 把这些 perceptron 的识别结果组合起来形成一个非线性的模型；理论上主种模型可以识别任意目标函数 (universal approximator)。

通常只需要一个 hidden layer 就足够解决大部分问题，再增加 hidden layer 的数量并不会带来多大性能提升。

而每一个 hidden layer 中的 unit 数量通常在 layer 的输入层的大小跟输出层的大小之间。

[Is a single layered ReLu network still a universal approximator](https://www.quora.com/Is-a-single-layered-ReLu-network-still-a-universal-approximator/answer/Conner-Davis-2)

[Why do neural networks need more than one hidden layer](https://www.quora.com/Why-do-neural-networks-need-more-than-one-hidden-layer)

[How to choose size of hidden layers](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
