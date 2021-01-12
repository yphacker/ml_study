# # coding=utf-8
# # author=yphacker
#
# def lgb_grid_search():
#     train_data = pd.read_csv('train.csv')  # 读取数据
#     y = train_data.pop('30').values  # 用pop方式将训练数据中的标签值y取出来，作为训练目标，这里的‘30’是标签的列名
#     col = train_data.columns
#     x = train_data[col].values  # 剩下的列作为训练数据
#     train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)  # 分训练集和验证集
#     train = lgb.Dataset(train_x, train_y)
#     valid = lgb.Dataset(valid_x, valid_y, reference=train)
#
#     parameters = {
#         'max_depth': [15, 20, 25, 30, 35],
#         'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#         'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#         'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#         'bagging_freq': [2, 4, 5, 6, 8],
#         'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
#         'lambda_l2': [0, 10, 15, 35, 40],
#         'cat_smooth': [1, 10, 15, 20, 35]
#     }
#     gbm = lgb.LGBMClassifier(boosting_type='gbdt',
#                              objective='binary',
#                              metric='auc',
#                              verbose=0,
#                              learning_rate=0.01,
#                              num_leaves=35,
#                              feature_fraction=0.8,
#                              bagging_fraction=0.9,
#                              bagging_freq=8,
#                              lambda_l1=0.6,
#                              lambda_l2=0)
#     # 有了gridsearch我们便不需要fit函数
#     gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
#     gsearch.fit(train_x, train_y)
#
#     print("Best score: %0.3f" % gsearch.best_score_)
#     print("Best parameters set:")
#     best_parameters = gsearch.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
#     # 调参1：提高准确率"：num_leaves, max_depth, learning_rate
#     # 调参2：降低过拟合 max_bin min_data_in_leaf
#     # 调参3：降低过拟合 正则化L1, L2
#     # 调参4：降低过拟合 数据抽样 列抽样
#     # 调参方向：处理过拟合（过拟合和准确率往往相反）
#     # 使用较小的 max_bin
#     # 使用较小的 num_leaves
#     # 使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
#     # 通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
#     # 通过设置 feature_fraction <1来使用特征抽样
#     # 使用更大的训练数据
#     # 使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
#     # 尝试 max_depth 来避免生成过深的树
#
# def xgb_grid_search():
#     train_data = pd.read_csv('train.csv')  # 读取数据
#     y = train_data.pop('30').values  # 用pop方式将训练数据中的标签值y取出来，作为训练目标，这里的‘30’是标签的列名
#     col = train_data.columns
#     x = train_data[col].values  # 剩下的列作为训练数据
#     train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)  # 分训练集和验证集
#     # 这里不需要Dmatrix
#
#     parameters = {
#         'max_depth': [5, 10, 15, 20, 25],
#         'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#         'n_estimators': [500, 1000, 2000, 3000, 5000],
#         'min_child_weight': [0, 2, 5, 10, 20],
#         'max_delta_step': [0, 0.2, 0.6, 1, 2],
#         'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
#         'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#         'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
#         'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
#         'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
#
#     }
#
#     xlf = xgb.XGBClassifier(max_depth=10,
#                             learning_rate=0.01,
#                             n_estimators=2000,
#                             silent=True,
#                             objective='binary:logistic',
#                             nthread=-1,
#                             gamma=0,
#                             min_child_weight=1,
#                             max_delta_step=0,
#                             subsample=0.85,
#                             colsample_bytree=0.7,
#                             colsample_bylevel=1,
#                             reg_alpha=0,
#                             reg_lambda=1,
#                             scale_pos_weight=1,
#                             seed=1440,
#                             missing=None)
#
#     # 有了gridsearch我们便不需要fit函数
#     gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
#     gsearch.fit(train_x, train_y)
#
#     print("Best score: %0.3f" % gsearch.best_score_)
#     print("Best parameters set:")
#     best_parameters = gsearch.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         print("\t%s: %r" % (param_name, best_parameters[param_name]))

# ||xgb|lgb|range|
# |叶子数|num_leaves|num_leaves|range(35,65,5)|
# |树深|max_depth|max_depth|range(3,10,2)|
# |样本抽样|subsample|bagging_fraction，subsample|[i/10.0 for i in range(6,10)]|
# |特征抽样|colsample_bytree|feature_fraction，colsample_bytree|[i/10.0 for i in range(6,10)]|
# |L1正则化|alpha，reg_alpha|ambda_l2，reg_alpha|[1e-5, 1e-2, 0.1, 1, 2,2.5,3]|
# |L2正则化|lambda，reg_lambda|lambda_l1，reg_lambda|[1e-5, 1e-2, 0.1, 1, 2,2.5,3]|