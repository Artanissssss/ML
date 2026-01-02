# 机器学习期末复习指南

## 📚 课程材料概览

本仓库包含以下8个主题的学习资料：

1. **Introduction** - 机器学习导论
2. **Statistical Learning** - 统计学习
3. **Supervised Learning** - 监督学习
4. **Neural Networks** - 神经网络
5. **Deep Learning** - 深度学习
6. **Clustering** - 聚类
7. **Density Estimation** - 密度估计
8. **Dimensionality Reduction** - 降维

---

## 🎯 核心知识点总结

### 1. 机器学习导论 (Introduction)

**关键概念：**
- **机器学习定义**：让计算机从数据中学习，无需明确编程
- **学习类型**：
  - 监督学习（Supervised Learning）
  - 无监督学习（Unsupervised Learning）
  - 强化学习（Reinforcement Learning）
  - 半监督学习（Semi-supervised Learning）
- **基本术语**：
  - 特征（Features）
  - 标签（Labels）
  - 训练集（Training Set）
  - 测试集（Test Set）
  - 验证集（Validation Set）

**重点理解：**
- 过拟合（Overfitting）vs 欠拟合（Underfitting）
- 偏差-方差权衡（Bias-Variance Tradeoff）
- 泛化能力（Generalization）

---

### 2. 统计学习 (Statistical Learning)

**关键概念：**
- **统计学习理论基础**
  - 期望风险（Expected Risk）
  - 经验风险（Empirical Risk）
  - 结构风险最小化（Structural Risk Minimization）
  
- **模型评估**：
  - 损失函数（Loss Function）
  - 0-1损失、平方损失、绝对损失
  - 交叉验证（Cross-Validation）
  - K折交叉验证（K-Fold CV）

- **正则化**：
  - L1正则化（Lasso）
  - L2正则化（Ridge）
  - 弹性网络（Elastic Net）

**重要公式：**
```
经验风险 = (1/N) Σ L(yi, f(xi))
结构风险 = 经验风险 + λ × 复杂度
```

---

### 3. 监督学习 (Supervised Learning)

**核心算法：**

**3.1 线性模型**
- **线性回归**（Linear Regression）
  - 最小二乘法（Ordinary Least Squares）
  - 正规方程：θ = (X^T X)^(-1) X^T y
  
- **逻辑回归**（Logistic Regression）
  - Sigmoid函数：σ(z) = 1/(1+e^(-z))
  - 交叉熵损失函数

**3.2 决策树与集成方法**
- **决策树**（Decision Tree）
  - 信息增益（Information Gain）
  - 基尼不纯度（Gini Impurity）
  - 剪枝（Pruning）

- **随机森林**（Random Forest）
  - Bootstrap聚合（Bagging）
  - 特征随机选择

- **提升方法**（Boosting）
  - AdaBoost
  - Gradient Boosting
  - XGBoost

**3.3 支持向量机 (SVM)**
- 最大间隔分类器
- 核技巧（Kernel Trick）
- 软间隔（Soft Margin）

**3.4 贝叶斯方法**
- 朴素贝叶斯（Naive Bayes）
- 贝叶斯定理：P(A|B) = P(B|A)P(A)/P(B)

**3.5 K近邻 (KNN)**
- 距离度量（欧式距离、曼哈顿距离）
- K值选择的影响

---

### 4. 神经网络 (Neural Networks)

**基础架构：**
- **感知机**（Perceptron）
- **多层感知机**（MLP）
- **激活函数**：
  - Sigmoid: σ(x) = 1/(1+e^(-x))
  - Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
  - ReLU: f(x) = max(0, x)
  - Leaky ReLU
  - Softmax（用于多分类）

**训练算法：**
- **反向传播**（Backpropagation）
  - 链式法则（Chain Rule）
  - 梯度下降（Gradient Descent）
  
- **优化器**：
  - SGD（随机梯度下降）
  - Momentum
  - Adam
  - RMSprop

**关键技巧：**
- 批量归一化（Batch Normalization）
- Dropout（防止过拟合）
- 权重初始化策略
- 学习率调度

---

### 5. 深度学习 (Deep Learning)

**主要架构：**

**5.1 卷积神经网络 (CNN)**
- 卷积层（Convolution Layer）
- 池化层（Pooling Layer）- Max Pooling, Average Pooling
- 全连接层（Fully Connected Layer）
- 经典架构：LeNet, AlexNet, VGG, ResNet, Inception

**5.2 循环神经网络 (RNN)**
- 标准RNN
- LSTM（长短期记忆网络）
- GRU（门控循环单元）
- 序列到序列（Seq2Seq）
- 注意力机制（Attention Mechanism）

**5.3 其他重要模型**
- 自编码器（Autoencoder）
- 变分自编码器（VAE）
- 生成对抗网络（GAN）
- Transformer架构

**训练技巧：**
- 迁移学习（Transfer Learning）
- 数据增强（Data Augmentation）
- 早停法（Early Stopping）
- 梯度裁剪（Gradient Clipping）

---

### 6. 聚类 (Clustering)

**主要算法：**

**6.1 K-means聚类**
- 算法步骤：初始化 → 分配 → 更新 → 重复
- K值选择：肘部法则（Elbow Method）
- 优缺点分析

**6.2 层次聚类**（Hierarchical Clustering）
- 凝聚式（Agglomerative）
- 分裂式（Divisive）
- 链接方法：单链接、全链接、平均链接

**6.3 基于密度的聚类**
- **DBSCAN**
  - 核心点、边界点、噪声点
  - ε（邻域半径）和MinPts参数

**6.4 高斯混合模型 (GMM)**
- EM算法（期望最大化）
- 与K-means的区别

**评估指标：**
- 轮廓系数（Silhouette Coefficient）
- Calinski-Harabasz指数
- Davies-Bouldin指数

---

### 7. 密度估计 (Density Estimation)

**核心方法：**

**7.1 参数方法**
- 最大似然估计（MLE）
- 高斯分布假设
- 多元高斯分布

**7.2 非参数方法**
- **核密度估计**（Kernel Density Estimation, KDE）
  - 核函数选择（高斯核、Epanechnikov核）
  - 带宽（Bandwidth）选择
  
- **直方图方法**
  - 箱宽选择

**7.3 混合模型**
- 高斯混合模型（GMM）
- EM算法详解
  - E步骤（期望）
  - M步骤（最大化）

**应用：**
- 异常检测（Anomaly Detection）
- 生成新样本
- 概率估计

---

### 8. 降维 (Dimensionality Reduction)

**线性降维方法：**

**8.1 主成分分析 (PCA)**
- 方差最大化
- 协方差矩阵的特征值分解
- 主成分选择（累计方差贡献率）
- 数据预处理：标准化

**8.2 线性判别分析 (LDA)**
- 类间方差最大化
- 类内方差最小化
- 与PCA的区别（监督 vs 无监督）

**8.3 奇异值分解 (SVD)**
- 矩阵分解：A = UΣV^T
- 与PCA的关系

**非线性降维方法：**

**8.4 流形学习**
- **t-SNE**（t-分布随机邻域嵌入）
  - 高维相似度 → 低维相似度
  - 困惑度（Perplexity）参数
  - 主要用于可视化

- **Isomap**（等距映射）
- **LLE**（局部线性嵌入）

**8.5 自编码器**
- 编码器-解码器结构
- 瓶颈层作为低维表示

**应用场景：**
- 数据可视化
- 特征提取
- 降噪
- 加速训练

---

## 📋 快速复习检查清单

### 考前必查知识点

- [ ] **基础概念**
  - [ ] 监督学习 vs 无监督学习的区别
  - [ ] 过拟合的原因和解决方法
  - [ ] 偏差-方差权衡
  - [ ] 训练集、验证集、测试集的作用

- [ ] **模型评估**
  - [ ] 分类指标：准确率、精确率、召回率、F1分数
  - [ ] 回归指标：MSE、RMSE、MAE、R²
  - [ ] 混淆矩阵（Confusion Matrix）
  - [ ] ROC曲线和AUC

- [ ] **监督学习算法**
  - [ ] 线性回归的正规方程和梯度下降
  - [ ] 逻辑回归和sigmoid函数
  - [ ] 决策树的分裂准则
  - [ ] SVM的核函数
  - [ ] 集成学习：Bagging vs Boosting

- [ ] **神经网络**
  - [ ] 反向传播算法原理
  - [ ] 常用激活函数及其特点
  - [ ] 梯度消失和梯度爆炸问题
  - [ ] 各种优化器的区别

- [ ] **深度学习**
  - [ ] CNN的基本组件和工作原理
  - [ ] RNN、LSTM的结构和应用
  - [ ] Dropout和Batch Normalization的作用
  - [ ] 迁移学习的概念

- [ ] **聚类算法**
  - [ ] K-means算法步骤
  - [ ] DBSCAN参数的含义
  - [ ] 层次聚类的链接方法
  - [ ] 聚类评估指标

- [ ] **降维技术**
  - [ ] PCA的数学原理
  - [ ] PCA vs LDA的区别
  - [ ] t-SNE的使用场景
  - [ ] 如何选择降维后的维数

- [ ] **密度估计**
  - [ ] 核密度估计的原理
  - [ ] EM算法的两个步骤
  - [ ] 高斯混合模型

---

## 💡 重点公式速查表

### 回归与分类

**线性回归损失函数（MSE）：**
```
J(θ) = (1/2m) Σ(hθ(x^(i)) - y^(i))²
```

**梯度下降更新规则：**
```
θj := θj - α ∂J(θ)/∂θj
```

**Sigmoid函数：**
```
σ(z) = 1/(1 + e^(-z))
```

**交叉熵损失：**
```
L = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### 神经网络

**ReLU激活函数：**
```
f(x) = max(0, x)
```

**Softmax：**
```
softmax(zi) = e^(zi) / Σe^(zj)
```

### 评估指标

**精确率（Precision）：**
```
P = TP / (TP + FP)
```

**召回率（Recall）：**
```
R = TP / (TP + FN)
```

**F1分数：**
```
F1 = 2 × (P × R) / (P + R)
```

### PCA

**协方差矩阵：**
```
Cov(X) = (1/n)X^T X  (数据已中心化)
```

**方差解释比例：**
```
比例 = λk / Σλi
```

### K-means

**目标函数：**
```
J = Σ Σ ||x^(i) - μk||²
```

---

## 🎓 学习建议

### 复习策略

1. **按主题复习**
   - 先浏览所有PPT，建立整体框架
   - 针对每个主题，理解核心概念
   - 记忆关键公式和算法步骤

2. **重点突破**
   - 神经网络和深度学习（通常是重点）
   - 监督学习的经典算法
   - 数学推导：反向传播、PCA

3. **动手实践**
   - 复习时尝试手推公式
   - 在纸上画出算法流程图
   - 模拟算法执行过程

4. **对比学习**
   - 类似算法的异同（如K-means vs GMM）
   - 不同场景的算法选择
   - 各算法的优缺点

### 考试技巧

1. **概念题**
   - 准确理解术语定义
   - 能够用自己的话解释概念
   - 举例说明应用场景

2. **计算题**
   - 熟练掌握基本公式
   - 注意矩阵维度
   - 检查计算结果的合理性

3. **算法题**
   - 清楚写出算法步骤
   - 标注每步的目的
   - 分析时间和空间复杂度

4. **应用题**
   - 理解问题场景
   - 选择合适的算法
   - 说明选择理由

### 时间分配建议

- **基础概念复习**：30%
- **核心算法理解**：40%
- **公式推导练习**：20%
- **综合应用题**：10%

---

## 📌 常见考点

### 高频考点

1. **监督学习**
   - 线性回归和逻辑回归的推导
   - 决策树的构建过程
   - SVM的间隔最大化原理

2. **神经网络**
   - 反向传播算法
   - 激活函数的选择
   - 过拟合的防止方法

3. **深度学习**
   - CNN的卷积和池化操作
   - RNN处理序列数据的原理
   - 各种正则化技术

4. **无监督学习**
   - K-means的迭代过程
   - PCA的数学推导
   - 聚类结果的评估

5. **理论基础**
   - 偏差-方差分解
   - 正则化的作用
   - 交叉验证方法

### 易错点提醒

- PCA需要对数据进行标准化
- K-means对初始值敏感
- 逻辑回归是分类算法，不是回归算法
- Dropout只在训练时使用，测试时不用
- t-SNE主要用于可视化，不用于降维后建模

---

## 🔗 资料组织

本仓库的8个文件按照机器学习的知识体系组织：

```
ML/
├── 1-Introduction.pptx              # 入门基础
├── 2-Statistical Learning.pptx      # 理论基础
├── 3-Supervised Learning.pptx       # 核心算法
├── 4-Neural Networks.pptx           # 神经网络
├── 5-Deep Learning.pdf              # 深度学习
├── 6-Clustering.pptx                # 聚类方法
├── 7-Density Estimation.pptx        # 密度估计
└── 8-Dimensionality Reduction.pptx  # 降维技术
```

建议按照以上顺序复习，循序渐进。

---

## ✅ 考前最后检查

考试前一天：
- [ ] 快速浏览所有PPT
- [ ] 重点复习标记的难点
- [ ] 默写核心公式
- [ ] 复习错题和易错点
- [ ] 保证充足睡眠

考试当天：
- [ ] 提前到达考场
- [ ] 携带必要文具和计算器
- [ ] 保持冷静，仔细审题
- [ ] 先做会做的题
- [ ] 预留检查时间

---

## 💪 祝你考试顺利！

记住：
- 理解 > 记忆
- 练习 > 阅读
- 系统 > 零散

加油！相信你一定能取得好成绩！🎉

---

*最后更新：2026年1月*
