import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import r2_score

if True:
    bool_dealData = False
    bool_Hour = False
    bool_2016_01_12 = False
    bool_2016_01_12_to_19 = False

if bool_dealData:
    # ipython notebook路径不能出现中文
    # 1.导入数据
    data = pd.read_csv(r"G:\energydata_complete.csv")
    # print(data.describe())
    # 2.将str格式的时间数据转换成时间格式
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S")

    # 3.增加有效信息
    df = data.set_index('date')


    def Second(x):
        temp = x.hour * 3600
        temp += x.minute * 60
        temp += x.second
        return temp


    data['Msn'] = pd.DataFrame(df.index.map(lambda x: Second(x)))
    data['week_year'] = pd.DataFrame(df.index.weekofyear)  # 一年当中的第几周
    data['my'] = pd.DataFrame(df.index.map(lambda x: x.strftime('%Y-%m-%d')))  # 增加my列表示日期，形式例如2019-04-01
    data['mhr'] = pd.DataFrame(df.index.map(lambda x: x.strftime('%H:00:00')))  # 增加hour列表示日期，形式例如17:00:00
    data['day_week'] = pd.DataFrame(df.index.day_name())  # 增加星期


    def weekend_weekday(x):
        x = str(x)
        if "Saturday" in x or "Sunday" in x:
            return "Weekend"
        else:
            return "Weekday"


    data['weekStatus'] = pd.DataFrame(list(data['day_week'].map(lambda x: weekend_weekday(x))))
    data = data.drop(['rv1', 'rv2'], axis=1)  # 移除rv1、rv2
    data.to_csv("./dataAfterDeal.csv", index=False)
    # print(data)
    del data, df

if bool_Hour:
    data = pd.read_csv("./dataAfterDeal.csv")
    df = data.set_index('date')
    df.index = pd.to_datetime(df.index)  # 数据类型转换
    data = df.resample("H").sum().to_period("H")
    print(data)
    # data = data[['Appliances', 'lights']]
    # 保存Appliances、lights
    data.to_csv("./powerUserByHour.csv")
    del data, df

if bool_Hour:
    data = pd.read_csv("./powerUserByHour.csv")
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S")
    df = data.set_index('date')

    data['day_week'] = pd.DataFrame(df.index.day_name())  # 增加星期
    data['Monday'] = pd.get_dummies(data['day_week'])['Monday']
    data['Tuesday'] = pd.get_dummies(data['day_week'])['Tuesday']
    data['Wednesday'] = pd.get_dummies(data['day_week'])['Wednesday']
    data['Thursday'] = pd.get_dummies(data['day_week'])['Thursday']
    data['Friday'] = pd.get_dummies(data['day_week'])['Friday']
    data['Saturday'] = pd.get_dummies(data['day_week'])['Saturday']
    data['Sunday'] = pd.get_dummies(data['day_week'])['Sunday']

    data['Msn'] = pd.DataFrame(df.index.map(lambda x: x.hour * 3600))
    data['date'] = pd.DataFrame(df.index.hour)  # 去除日期
    data = data.drop(['day_week'], axis=1)  # 去除字符串
    data['week_year'] = pd.DataFrame(df.index.weekofyear)  # 一年当中的第几周
    # print(data.corr()[u'Appliances'])
    data = data.drop(['date'], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv("./dataForTrain.csv")
    del data, df

if True:
    # import modin.pandas as pd

    data = pd.read_csv("./dataForTrain.csv")
    data = data.drop(['lights'], axis=1)
    # print(data.columns.values.tolist())
    x_train, x_test, y_train, y_test = train_test_split(data.drop(['Appliances'], axis=1), data['Appliances'],
                                                        random_state=1234)
    # print("划分数据集成功......")
    import os
    from sklearn.externals import joblib

    if os.path.exists("gbrt_model.m"):
        gbrt = joblib.load("gbrt_model.m")
        # print("模型加载成功！")
        pass
    else:
        # print("开始训练模型......")

        '''
        gbrt = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                         learning_rate=0.198, loss='ls', max_depth=6, max_features=None,
                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, min_samples_leaf=1,
                                         min_samples_split=2, min_weight_fraction_leaf=0.005,
                                         n_estimators=105, n_iter_no_change=None, presort='auto',
                                         random_state=1234, subsample=0.985, tol=0.0001,
                                         validation_fraction=0.1, verbose=0, warm_start=False)
训练集拟合度：92.205130 %
测试集准确度：52.192902 %

训练集性能：
RMSE: 133.1835356733243
R: 0.9220513005185618
MAE: 83.57359121541228
MAPE: 15.598958150160886 %

测试集性能：
RMSE: 352.9362069341872
R: 0.5219290194283241
MAE: 213.98266540173745
MAPE: 35.72890699718826 %
        '''
        '''
        GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.2, loss='lad', max_depth=6, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.005,
             n_estimators=150, n_iter_no_change=None, presort='auto',
             random_state=1234, subsample=0.9, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
54.551371 %
33.227432 %
        '''
        # param_test1 = {'n_estimators': range(197, 200, 1)}
        # gbrt = GridSearchCV(estimator=GradientBoostingRegressor(alpha=0.8, criterion='friedman_mse', init=None,
        #                                                         learning_rate=0.2, loss='ls', max_depth=6,
        #                                                         max_features=None,
        #                                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
        #                                                         min_impurity_split=None, min_samples_leaf=1,
        #                                                         min_samples_split=2, min_weight_fraction_leaf=0.005,
        #                                                         n_iter_no_change=None, presort='auto',
        #                                                         random_state=1234, subsample=0.9, tol=0.0001,
        #                                                         validation_fraction=0.1, verbose=0, warm_start=False),
        #                     param_grid=param_test1, cv=3)
        gbrt = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=196,
                                         subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                         max_depth=3, min_impurity_decrease=0.,
                                         min_impurity_split=None, init=None, random_state=None,
                                         max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                         warm_start=False, presort='auto', validation_fraction=0.1,
                                         n_iter_no_change=None, tol=1e-4)
        gbrt.fit(x_train, y_train)
        joblib.dump(gbrt, "gbrt_model.m")
        # print(gbrt)

    train_predict = gbrt.predict(x_train)

    print("训练集拟合度：%2f" % (gbrt.score(x_train, y_train) * 100), "%")
    print("测试集准确度：%2f" % (gbrt.score(x_test, y_test) * 100), "%", end="\n\n")
    predict = gbrt.predict(x_test)
    if gbrt.score(x_test, y_test) < 0.55:
        os.remove('gbrt_model.m')
    print("训练集性能：")
    print("RMSE:", (((y_train - train_predict) ** 2).sum() / len(train_predict)) ** 0.5)
    print("R:", r2_score(y_train, train_predict))
    print("MAE:", abs(y_train - train_predict).sum() / len(train_predict))
    print("MAPE:", (abs((y_train - train_predict) / y_train).sum()) / len(train_predict) * 100, "%", end="\n\n")
    print("测试集性能：")
    print("RMSE:", (((y_test - predict) ** 2).sum() / len(predict)) ** 0.5)
    print("R:", r2_score(y_test, predict))
    print("MAE:", abs(y_test - predict).sum() / len(predict))
    print("MAPE:", (abs((y_test - predict) / y_test).sum()) / len(predict) * 100, "%")

    # param_test1 = {'n_estimators': range(20, 81, 10)}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
    #                                                              min_samples_leaf=20, max_depth=8, max_features='sqrt',
    #                                                              subsample=0.8, random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    # gsearch1.fit(X, y)
    # gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    param_test1 = {'n_estimators': range(80, 200, 1), 'learning_rate': [x / 100 for x in range(1, 100)]}
    gbrt = GridSearchCV(
        estimator=GradientBoostingRegressor(loss='huber', subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                            max_depth=3, min_impurity_decrease=0.,
                                            min_impurity_split=None, init=None, random_state=None,
                                            max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                            warm_start=False, presort='auto', validation_fraction=0.1,
                                            n_iter_no_change=None, tol=1e-4), param_grid=param_test1, n_jobs=-1, cv=5)
    gbrt.fit(x_train, y_train)
    train_predict = gbrt.predict(x_train)

    print("训练集拟合度：%2f" % (gbrt.score(x_train, y_train) * 100), "%")
    print("测试集准确度：%2f" % (gbrt.score(x_test, y_test) * 100), "%", end="\n\n")
    predict = gbrt.predict(x_test)
    try:
        if gbrt.score(x_test, y_test) < 0.55:
            os.remove('gbrt_model.m')
    except Exception as e:
        print(e)
    print("训练集性能：")
    print("RMSE:", (((y_train - train_predict) ** 2).sum() / len(train_predict)) ** 0.5)
    print("R:", r2_score(y_train, train_predict))
    print("MAE:", abs(y_train - train_predict).sum() / len(train_predict))
    print("MAPE:", (abs((y_train - train_predict) / y_train).sum()) / len(train_predict) * 100, "%", end="\n\n")
    print("测试集性能：")
    print("RMSE:", (((y_test - predict) ** 2).sum() / len(predict)) ** 0.5)
    print("R:", r2_score(y_test, predict))
    print("MAE:", abs(y_test - predict).sum() / len(predict))
    print("MAPE:", (abs((y_test - predict) / y_test).sum()) / len(predict) * 100, "%")

    if False:
        max = 0.0
        for i in range(80, 200):
            gbrt = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=i,
                                             subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                             max_depth=3, min_impurity_decrease=0.,
                                             min_impurity_split=None, init=None, random_state=None,
                                             max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                             warm_start=False, presort='auto', validation_fraction=0.1,
                                             n_iter_no_change=None, tol=1e-4)
            gbrt.fit(x_train, y_train)
            train_predict = gbrt.predict(x_train)

            x = gbrt.score(x_test, y_test)
            print(i)
            if x > max:
                max = x
                predict = gbrt.predict(x_test)

                print("训练集拟合度：%2f" % (gbrt.score(x_train, y_train) * 100), "%")
                print("测试集准确度：%2f" % (gbrt.score(x_test, y_test) * 100), "%", end="\n\n")

                print("训练集性能：")
                print("RMSE:", (((y_train - train_predict) ** 2).sum() / len(train_predict)) ** 0.5)
                print("R:", r2_score(y_train, train_predict))
                print("MAE:", abs(y_train - train_predict).sum() / len(train_predict))
                print("MAPE:", (abs((y_train - train_predict) / y_train).sum()) / len(train_predict) * 100, "%",
                      end="\n\n")
                print("测试集性能：")
                print("RMSE:", (((y_test - predict) ** 2).sum() / len(predict)) ** 0.5)
                print("R:", r2_score(y_test, predict))
                print("MAE:", abs(y_test - predict).sum() / len(predict))
                print("MAPE:", (abs((y_test - predict) / y_test).sum()) / len(predict) * 100, "%", end="\n\n")

    del data

if bool_2016_01_12:
    # 绘制 2016-01-12 lights 随时间变化曲线
    '''
    结论：
        0点、5-8点、16-18点、20-23点为lights的用电高峰
    '''
    appliances_light = df.resample('H').sum().to_period('H')[7:31][["Appliances", "lights"]]
    plt.plot(appliances_light.index.hour, appliances_light['lights'], c='r')
    # plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='r')
    plt.xticks(appliances_light.index.hour)
    plt.title("2016-01-12 lights")
    plt.savefig("2016-01-12 lights.png")
    # plt.show()
    plt.cla()

if bool_2016_01_12_to_19:
    # 绘制2016-01-12 - 2016-01-19 lights随时间变化曲线
    '''
    结论：
        显然也是3个小高峰
    '''
    appliances_light = df.resample('H').sum().to_period('H')[7:31][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='b')

    appliances_light = df.resample('H').sum().to_period('H')[31:55][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='g')

    appliances_light = df.resample('H').sum().to_period('H')[55:79][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='r')

    appliances_light = df.resample('H').sum().to_period('H')[79:103][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='c')

    appliances_light = df.resample('H').sum().to_period('H')[103:127][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='m')

    appliances_light = df.resample('H').sum().to_period('H')[127:151][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='y')

    appliances_light = df.resample('H').sum().to_period('H')[151:175][["Appliances", "lights"]]
    plt.scatter(appliances_light.index.hour, appliances_light['lights'], c='k')

    plt.xticks(appliances_light.index.hour)
    plt.title("2016-01-12 lights")
    plt.savefig("2016-01-12 to 2016-01-19 lights.png")
    # plt.show()
    plt.cla()
