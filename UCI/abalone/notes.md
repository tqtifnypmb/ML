## Abalone Notes

[Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone) 
Notes

---

### Feature selection

#### Variance

一个变量的变化越小，它所包含的信息量就越少，一个方差接近 0 的变量，基乎就是一个常量了。

#### Mutual information

MI 衡量两个变量之间的相关程度，相当于 D(p(x,y) || p(x)p(y))，可以解释为用 p(x,y) 取代 p(x)p(y) 所能获得的信息量，或者 p(x,y) 跟 p(x)p(y) 之间“距离”。所以两个变量的越不独立，它们的 MI 值越高。

---

### Metrics for multi-class classification

#### accuracy score

accuracy score 支持 muti-class 问题，但是在 class 不均匀时指示意义不大。

#### precision

precision score 反映模型作出的 positive 预测的准确度。

#### recall

recall score 反映模型发现数据中 positive 样本的能力。

---

### Imbalance multi-class problem

某些类型的样本量太少导致基于 K-MEANS 的 oversampling 无法进行，需要进行 RandomOversampling。
