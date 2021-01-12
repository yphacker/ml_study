# XGBoost

## 文档
[英文文档](https://xgboost.readthedocs.io/en/latest/)
[中文文档](https://xgboost.apachecn.org/#/)

## xgb
||objective|eval_metric|
|二分类|binary:logistic|auc|
|多分类|multi:softmax/multi:softprob|mlogloss|
|回归|reg:linear|mae|

1.通用参数
booster：我们有两种参数选择，gbtree和gblinear。gbtree是采用树的结构来运行数据，而gblinear是基于线性模型。
silent：静默模式，为1时模型运行不输出。
nthread: 使用线程数，一般我们设置成-1,使用所有线程。如果有需要，我们设置成多少就是用多少线程。
2.Booster参数
n_estimator: 也作num_boosting_rounds

这是生成的最大树的数目，也是最大的迭代次数。

learning_rate: 有时也叫作eta，系统默认值为0.3,。

每一步迭代的步长，很重要。太大了运行准确率不高，太小了运行速度慢。我们一般使用比默认值小一点，0.1左右就很好。

gamma：系统默认为0,我们也常用0。

在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。因为gamma值越大的时候，损失函数下降更多才可以分裂节点。所以树生成的时候更不容易分裂节点。范围: [0,∞]

subsample：系统默认为1。

这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1，0.5代表平均采样，防止过拟合. 范围: (0,1]，注意不可取0

colsample_bytree：系统默认值为1。我们一般设置成0.8左右。

用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1范围: (0,1]

colsample_bylevel：默认为1,我们也设置为1.

这个就相比于前一个更加细致了，它指的是每棵树每次节点分裂的时候列采样的比例

max_depth： 系统默认值为6

我们常用3-10之间的数字。这个值为树的最大深度。这个值是用来控制过拟合的。max_depth越大，模型学习的更加具体。设置为0代表没有限制，范围: [0,∞]

max_delta_step：默认0,我们常用0.

这个参数限制了每棵树权重改变的最大步长，如果这个参数的值为0,则意味着没有约束。如果他被赋予了某一个正值，则是这个算法更加保守。通常，这个参数我们不需要设置，但是当个类别的样本极不平衡的时候，这个参数对逻辑回归优化器是很有帮助的。

lambda:也称reg_lambda,默认值为0。

权重的L2正则化项。(和Ridge regression类似)。这个参数是用来控制XGBoost的正则化部分的。这个参数在减少过拟合上很有帮助。

alpha:也称reg_alpha默认为0,
权重的L1正则化项。(和Lasso regression类似)。 可以应用在很高维度的情况下，使得算法的速度更快。

scale_pos_weight：默认为1
在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。通常可以将其设置为负样本的数目与正样本数目的比值。

3.学习目标参数
objective [缺省值=reg:linear]
reg:linear– 线性回归
reg:logistic – 逻辑回归
binary:logistic – 二分类逻辑回归，输出为概率
binary:logitraw – 二分类逻辑回归，输出的结果为wTx
count:poisson – 计数问题的poisson回归，输出结果为poisson分布。在poisson回归中，max_delta_step的缺省值为0.7 (used to safeguard optimization)
multi:softmax – 设置 XGBoost 使用softmax目标函数做多分类，需要设置参数num_class（类别个数）
multi:softprob – 如同softmax，但是输出结果为ndata*nclass的向量，其中的值是每个数据分为每个类的概率。
eval_metric [缺省值=通过目标函数选择]
rmse: 均方根误差
mae: 平均绝对值误差
logloss: negative log-likelihood
error: 二分类错误率。其值通过错误分类数目与全部分类数目比值得到。对于预测，预测值大于0.5被认为是正类，其它归为负类。 error@t: 不同的划分阈值可以通过 ‘t’进行设置
merror: 多分类错误率，计算公式为(wrong cases)/(all cases)
mlogloss: 多分类log损失
auc: 曲线下的面积
ndcg: Normalized Discounted Cumulative Gain
map: 平均正确率
一般来说，我们都会使用xgboost.train(params, dtrain)函数来训练我们的模型。这里的params指的是booster参数。
