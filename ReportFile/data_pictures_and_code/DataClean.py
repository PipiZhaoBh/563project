import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def miss_data_count(data: pd.DataFrame) -> pd.Series:
    miss_data = data.isnull().sum()  # find NA data count for each column
    return miss_data[miss_data > 0].sort_values(ascending=True)  # order miss data count from low to high


def general_linear_regression(y, x):
    # y = a_1 * x_1 + a_2 * x_2 + ... + b
    # Number of a_i equals to number of the column of x

    # use sklearn to do a linear regression on x y
    linear_regression = LinearRegression()
    linear_regression.fit(x, y)
    plt.plot(linear_regression.predict(x.values), y.values, '.')
    plt.show()

    return linear_regression


def impute_LotFrontage(data: pd.DataFrame):
    # impute column LotFrontage as an example
    # LotFrontage = a * (sqrt(LotArea) + sqrt(1stFlrSF) + TotRmsAbvGrd) + b
    xy = data[['LotFrontage', 'LotArea', '1stFlrSF', 'TotRmsAbvGrd']]
    xy.dropna(how='any', axis=0, inplace=True)  # Drop rows with empty elements
    x = xy.loc[:, ['LotArea', '1stFlrSF', 'TotRmsAbvGrd']].applymap(lambda x : np.sqrt(x))
    y = xy.loc[:, ['LotFrontage']]
    linear_regression = general_linear_regression(y, x)

    def impute_lotfrontage(df):
        if pd.isna(df['LotFrontage']):
            x_1 = np.array([df['LotArea'], df['1stFlrSF'], df['1stFlrSF']])
            x_2 = np.sqrt(x_1)
            y_1 = linear_regression.predict(x_2.reshape(1, -1))
            return y_1.ravel()[0]
        else:
            return df['LotFrontage']

    data.loc[:, ['LotFrontage']] = data.apply(impute_lotfrontage, axis=1)
    return data


def impute_MasVnrArea(data: pd.DataFrame):
    # impute column LotFrontage as an example
    # LotFrontage = a * (GarageArea
    y_label = 'MasVnrArea'
    # x_label = ['YearBuilt', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
    x_label = ['OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces', 'FullBath']
    xy = data[[y_label] + x_label]
    xy.dropna(how='any', axis=0, inplace=True)  # Drop rows with empty elements
    xy = xy[xy['MasVnrArea'] != 0]
    x = xy.loc[:, x_label]
    y = xy.loc[:, ['MasVnrArea']]
    # x = x.applymap(np.abs)
    # x = x.applymap(lambda z: z + 1)
    # x = x.applymap(np.log)
    linear_regression = general_linear_regression(y, x)

    def impute_masvnrarea(df):
        if pd.isna(df[y_label]):
            x_1 = np.array([df[x_label].values]).astype('float').ravel()
            y_1 = linear_regression.predict(x_1.reshape(1, -1))
            # print(x_1, y_1)
            return y_1.ravel()[0]
            # a = linear_regression.predict(np.array(df[x_label]).reshape(1, -1))
            # return a
        else:
            return df[y_label]

    data.loc[:, [y_label]] = data.apply(impute_masvnrarea, axis=1)
    return data


def impute_by_linear_regression(data: pd.DataFrame, x_label: str, y_label: list) -> pd.DataFrame:
    xy = data[[y_label] + x_label]
    xy.dropna(how='any', axis=0, inplace=True)  # Drop rows with empty elements
    x = xy.loc[:, x_label].applymap(lambda z: np.sqrt(z))
    y = xy.loc[:, [y_label]]
    linear_regression = general_linear_regression(y, x)

    def impute_x(df):
        if pd.isna(df[y_label]):
            return linear_regression.predict(np.array(df[x_label]).reshape(1, -1))
        else:
            return df[y_label]

    data.loc[:, [y_label]] = data.apply(impute_x, axis=1)
    return data


def greedy_target_encoding(data: pd.DataFrame, column_name: str, a: int) -> pd.DataFrame:
    # A common setting for p is the average target value in the dataset
    # p is a experience parameter
    # print(column_name)
    train_data_path = r'G:\563project\train.csv'
    train_data = pd.read_csv(train_data_path)
    train_data = train_data.loc[:, [column_name, 'SalePrice']]
    train_data.fillna('It is Na', inplace=True)
    train_data_group = train_data.groupby(column_name).agg('sum')
    sale_price_sum = train_data_group.sum().values.ravel()
    train_data_group = (train_data_group + a * sale_price_sum) / train_data_group + a
    dic = {}
    dic = dict(zip(list(train_data_group.index), list(train_data_group.SalePrice)))
    print(column_name, dic)
    data[column_name] = data[column_name].fillna('It is Na').map(lambda x: dic[x])
    return data



def clean_data(path: str):
    data = read_data(path)
    miss_data = miss_data_count(data)
    print(miss_data)

    # Impute GarageQual and apply greedy target encoding with reasonable hyper parameters.
    data = greedy_target_encoding(data, 'GarageQual', 0.5)

    # impute features those NA means don't have a xxx
    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageFinish",
             "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
             "MasVnrType"]
    for col in cols1:
        data[col].fillna("It is Na", inplace=True)

    # impute features those NA means don't have a xxx so that the features should be 0
    cols2 = ["BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea", "BsmtFullBath", "BsmtHalfBath"]
    for col in cols2:
        data[col].fillna(0, inplace=True)

    # Apply greedy target encoding with reasonable hyper parameters on col1 and col2.
    for col in cols1:
        data = greedy_target_encoding(data, col, 0.5)

    # impute column LotFrontage as an example
    data = impute_LotFrontage(data)
    # impute column MasVnrArea
    data = impute_MasVnrArea(data)
    # cols = cols1 + cols2 + ['GarageQual', 'MasVnrArea', 'LotFrontage']
    # data = data.loc[:, cols]
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(np.mean(data['GarageYrBlt'].dropna().values))
    return data

def pca_redude_dimension(df, n):
    pca = PCA(n_components=n)
    pca.fit(df)
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_)
    df_new = pca.transform(df)
    df_new = pd.DataFrame(df_new)
    df_new.to_csv(r'new_feature.csv')
    return pca

def coumpte_variance_importance(data):
    column_count = len(data_num.columns)
    x_total = data.iloc[:, :column_count - 1]
    y_total = data.iloc[:, column_count - 1]
    reg2 = ensemble.GradientBoostingRegressor(**params)
    reg2.fit(x_total, y_total)
    y_reg = reg2.predict(x_total)
    x_total['predict'] = y_reg
    x_total.corr()
    corr = np.abs(x_total.corr().predict.values)
    labels = list(x_total.columns)
    sorting = corr.argsort()
    labels_sort = [labels[i] for i in sorting]
    corr_sort = (np.sort(corr))
    print(corr_sort)
    plt.barh(range(len(corr_sort)), corr_sort, tick_label=labels_sort)
    plt.show()

def predict(reg, path):



    pass


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('TkAgg')
    path = r'G:/563project/train.csv'

    data = clean_data(path)
    data_num = data._get_numeric_data()
    # pca_redude_dimension(data_num, 3)
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
    }
    column_count = len(data_num.columns)
    # coumpte_variance_importance(data_num)
    x_data = data_num.iloc[:, :column_count - 1]
    y_data = data_num.iloc[:, column_count - 1]
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(x_data, y_data)
    # x_test = data_num.iloc[1000:, :column_count - 1]
    # y_test = data_num.iloc[1000:, column_count - 1]
    # y_reg = reg.predict(x_test)
    #
    #
    # plt.plot(y_reg, y_test.values, '.')
    # plt.plot(np.arange(900000), np.arange(900000))
    # plt.show()

    test_path = r'G:/563project/test.csv'
    test_data = clean_data(test_path)
    using_cols = list(x_data.columns)
    x_test = test_data.loc[:, using_cols]
    y_test = reg.predict(x_test)
    pd.DataFrame(y_test).to_csv('result.csv')
    # feature_importance = reg.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
    # pos = np.arange(sorted_idx.shape[0]) + 0.5
    # fig = plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.barh(pos, feature_importance[sorted_idx], align="center")
    # plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
    # plt.title("Feature Importance (MDI)")
    #
    # result = permutation_importance(
    #     reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    # )
    # sorted_idx = result.importances_mean.argsort()
    # plt.subplot(1, 2, 2)
    # plt.boxplot(
    #     result.importances[sorted_idx].T,
    #     vert=False,
    #     labels=np.array(diabetes.feature_names)[sorted_idx],
    # )
    # plt.title("Permutation Importance (test set)")
    # fig.tight_layout()
    # plt.show()
    #
    #
