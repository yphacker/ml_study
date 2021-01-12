# LightGBM

## 文档
[中文文档](https://lightgbm.apachecn.org/#/)
[英文文档](https://lightgbm.readthedocs.io/en/latest/)

### lgb
||objective|metric|
|二分类|||
|多分类|multiclass|multi_logloss/custom_feval|
|回归|regression|mae|

1.核心参数
boosting：也称boost，boosting_type.默认是gbdt。

LGB里面的boosting参数要比xgb多不少，我们有传统的gbdt，也有rf，dart，doss，最后两种不太深入理解，但是试过，还是gbdt的效果比较经典稳定

gbdt, 传统的梯度提升决策树
rf, Random Forest (随机森林)
dart, Dropouts meet Multiple Additive Regression Trees
goss, Gradient-based One-Side Sampling (基于梯度的单侧采样)
num_thread:也称作num_thread,nthread.指定线程的个数。

这里官方文档提到，数字设置成cpu内核数比线程数训练效更快(考虑到现在cpu大多超线程)。并行学习不应该设置成全部线程，这反而使得训练速度不佳。

application：默认为regression。，也称objective， app这里指的是任务目标

regression
regression_l2, L2 loss, alias=regression, mean_squared_error, mse
regression_l1, L1 loss, alias=mean_absolute_error, mae
huber, Huber loss
fair, Fair loss
poisson, Poisson regression
quantile, Quantile regression
quantile_l2, 类似于 quantile, 但是使用了 L2 loss
binary, binary log loss classification application
multi-class classification
multiclass, softmax 目标函数, 应该设置好 num_class
multiclassova, One-vs-All 二分类目标函数, 应该设置好 num_class
cross-entropy application
xentropy, 目标函数为 cross-entropy (同时有可选择的线性权重), alias=cross_entropy
xentlambda, 替代参数化的 cross-entropy, alias=cross_entropy_lambda
标签是 [0, 1] 间隔内的任意值
lambdarank, lambdarank application
在 lambdarank 任务中标签应该为 int type, 数值越大代表相关性越高 (e.g. 0:bad, 1:fair, 2:good, 3:perfect)
label_gain 可以被用来设置 int 标签的增益 (权重)
valid:验证集选用，也称test，valid_data, test_data.支持多验证集，以,分割

learning_rate:也称shrinkage_rate,梯度下降的步长。默认设置成0.1,我们一般设置成0.05-0.2之间

num_leaves:也称num_leaf,新版lgb将这个默认值改成31,这代表的是一棵树上的叶子数

num_iterations：也称num_iteration, num_tree, num_trees, num_round, num_rounds,num_boost_round。迭代次数

device：default=cpu, options=cpu, gpu

为树学习选择设备, 你可以使用 GPU 来获得更快的学习速度
Note: 建议使用较小的 max_bin (e.g. 63) 来获得更快的速度
Note: 为了加快学习速度, GPU 默认使用32位浮点数来求和. 你可以设置 gpu_use_dp=true 来启用64位浮点数, 但是它会使训练速度降低
Note: 请参考 安装指南 来构建 GPU 版本
2.学习控制参数
max_depth
default=-1, type=int限制树模型的最大深度. 这可以在 #data 小的情况下防止过拟合. 树仍然可以通过 leaf-wise 生长.
< 0 意味着没有限制.
feature_fraction：default=1.0, type=double, 0.0 < feature_fraction < 1.0, 也称sub_feature, colsample_bytree

如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征. 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征
可以用来加速训练
可以用来处理过拟合
bagging_fraction：default=1.0, type=double, 0.0 < bagging_fraction < 1.0, 也称sub_row, subsample

类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
可以用来加速训练
可以用来处理过拟合
Note: 为了启用 bagging, bagging_freq 应该设置为非零值
bagging_freq： default=0, type=int, 也称subsample_freq

bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
Note: 为了启用 bagging, bagging_fraction 设置适当
lambda_l1:默认为0,也称reg_alpha，表示的是L1正则化,double类型

lambda_l2:默认为0,也称reg_lambda，表示的是L2正则化，double类型

cat_smooth： default=10, type=double

用于分类特征
这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
min_data_in_leaf , 默认为20。 也称min_data_per_leaf , min_data, min_child_samples。
一个叶子上数据的最小数量。可以用来处理过拟合。

min_sum_hessian_in_leaf, default=1e-3, 也称min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight。

一个叶子上的最小 hessian 和. 类似于 min_data_in_leaf, 可以用来处理过拟合.
子节点所需的样本权重和(hessian)的最小阈值，若是基学习器切分后得到的叶节点中样本权重和低于该阈值则不会进一步切分，在线性模型中该值就对应每个节点的最小样本数，该值越大模型的学习约保守，同样用于防止模型过拟合
early_stopping_round, 默认为0, type=int, 也称early_stopping_rounds, early_stopping。
如果一个验证集的度量在 early_stopping_round 循环中没有提升, 将停止训练、

min_split_gain, 默认为0, type=double, 也称min_gain_to_split`。执行切分的最小增益。

max_bin：最大直方图数目，默认为255，工具箱的最大数特征值决定了容量 工具箱的最小数特征值可能会降低训练的准确性, 但是可能会增加一些一般的影响（处理过拟合，越大越容易过拟合）。

针对直方图算法tree_method=hist时，用来控制将连续值特征离散化为多个直方图的直方图数目。
LightGBM 将根据 max_bin 自动压缩内存。 例如, 如果 maxbin=255, 那么 LightGBM 将使用 uint8t 的特性值。
12.subsample_for_bin
bin_construct_sample_cnt, 默认为200000, 也称subsample_for_bin。用来构建直方图的数据的数量。
3.度量函数
metric： default={l2 for regression}, {binary_logloss for binary classification}, {ndcg for lambdarank}, type=multi-enum, options=l1, l2, ndcg, auc, binary_logloss, binary_error …
l1, absolute loss, alias=mean_absolute_error, mae
l2, square loss, alias=mean_squared_error, mse
l2_root, root square loss, alias=root_mean_squared_error, rmse
quantile, Quantile regression
huber, Huber loss
fair, Fair loss
poisson, Poisson regression
ndcg, NDCG
map, MAP
auc, AUC
binary_logloss, log loss
binary_error, 样本: 0 的正确分类, 1 错误分类
multi_logloss, mulit-class 损失日志分类
multi_error, error rate for mulit-class 出错率分类
xentropy, cross-entropy (与可选的线性权重), alias=cross_entropy
xentlambda, “intensity-weighted” 交叉熵, alias=cross_entropy_lambda
kldiv, Kullback-Leibler divergence, alias=kullback_leibler
支持多指标, 使用 , 分隔
总的来说，我还是觉得LightGBM比XGBoost用法上差距不大。参数也有很多重叠的地方。很多XGBoost的核心原理放在LightGBM上同样适用。 同样的，Lgb也是有train()函数和LGBClassifier()与LGBRegressor()函数。后两个主要是为了更加贴合sklearn的用法，这一点和XGBoost一样。

