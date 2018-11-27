import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error


def print_line(title):
    print("*" * 30 + " {} ".format(title) + "*" * 30)


def preprocess():
    # 获取数据
    boston = load_boston()
    print("feature_names:")
    print(boston.feature_names)
    print("data:")
    print(boston.data[:5])
    print("target:")
    print(boston.target[:5])

    # 划分训练集测试集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25)

    # 线性回归需要对特征数据和目标数据进行标准化处理
    # 特征数据标准化
    std_x = StandardScaler()
    std_x.fit(x_train)
    x_train = std_x.transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标数据标准化
    # StandardScaler的transform要求传入二维数据，故此处reshape
    y_train = y_train.reshape(-1, 1)

    std_y = StandardScaler()
    std_y.fit(y_train)
    y_train = std_y.transform(y_train)

    return x_train, x_test, y_train, y_test, std_y


def predict(model, x_test, y_test, std_y):
    # 回归系数
    print("回归系数：")
    print(model.coef_)

    # 预测结果
    print("真实值：")
    print(y_test[:10])
    y_predict = model.predict(x_test)
    y_predict = std_y.inverse_transform(y_predict)
    print("预测值：")
    print(np.around(y_predict[:10].flatten(), decimals=1))

    print("均方误差：")
    print(mean_squared_error(y_test, y_predict))


def linear(x_train, x_test, y_train, y_test, std_y):
    model = LinearRegression()
    model.fit(x_train, y_train)
    predict(model, x_test, y_test, std_y)


def sgd(x_train, x_test, y_train, y_test, std_y):
    # max_iter: 最大迭代次数
    model = SGDRegressor(max_iter=10)
    model.fit(x_train, y_train.ravel())
    predict(model, x_test, y_test, std_y)


def ridge(x_train, x_test, y_train, y_test, std_y):
    # alpha: 正则化力度
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train.ravel())
    predict(model, x_test, y_test, std_y)


if __name__ == '__main__':
    print_line("数据集")
    x_train, x_test, y_train, y_test, std_y = preprocess()

    print_line("线性回归：正规方程法")
    linear(x_train, x_test, y_train, y_test, std_y)

    print_line("线性回归：梯度下降法")
    sgd(x_train, x_test, y_train, y_test, std_y)

    print_line("岭回归")
    ridge(x_train, x_test, y_train, y_test, std_y)
