from utils import db_connect
engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

# We define the functions to do the EDA

def collect_data(raw_csv : str):
    _data = pd.read_csv(raw_csv)
    return _data

def save_data(data : pd.DataFrame, path : str):
    data.to_csv(path, index = False)

def delete_duplicated_values(data : pd.DataFrame):
    if data['name'].duplicated().sum() or data['host_id'].duplicated().sum() or data['id'].duplicated().sum():
        _clear_data = data.drop_duplicates()
    return _clear_data

def delete_irrelevant_information(data : pd.DataFrame):
    _clean_data = data.drop(['id', 'host_id', 'name', 'host_name', 'availability_365', 'calculated_host_listings_count', 'neighbourhood', 'latitude', 'longitude', 'last_review'], axis=1)
    return _clean_data

def categorical_analysis(data : pd.DataFrame):
    _fig, _axis = plt.subplots(2, 3, figsize=(10, 7))

    sns.histplot(ax = _axis[0,0], data = data, x = "host_id")
    sns.histplot(ax = _axis[0,1], data = data, x = "neighbourhood_group").set_xticks([])
    sns.histplot(ax = _axis[0,2], data = data, x = "neighbourhood").set_xticks([])
    sns.histplot(ax = _axis[1,0], data = data, x = "room_type")
    sns.histplot(ax = _axis[1,1], data = data, x = "availability_365")
    _fig.delaxes(_axis[1, 2])
    plt.tight_layout()
    plt.show()
    _fig.savefig("categorical_analysis.jpg")

def numerical_analysis(data : pd.DataFrame):
    _fig, _axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

    sns.histplot(ax = _axis[0, 0], data = data, x = "price")
    sns.boxplot(ax = _axis[1, 0], data = data, x = "price")

    sns.histplot(ax = _axis[0, 1], data = data, x = "minimum_nights").set_xlim(0, 200)
    sns.boxplot(ax = _axis[1, 1], data = data, x = "minimum_nights")

    sns.histplot(ax = _axis[2, 0], data = data, x = "number_of_reviews")
    sns.boxplot(ax = _axis[3, 0], data = data, x = "number_of_reviews")

    sns.histplot(ax = _axis[2,1], data = data, x = "calculated_host_listings_count")
    sns.boxplot(ax = _axis[3, 1], data = data, x = "calculated_host_listings_count")
    plt.tight_layout()
    plt.show()

    _fig.savefig("numerical_analysis.jpg")

def correlation_analysis(data : pd.DataFrame):
    data["room_type"] = pd.factorize(data["room_type"])[0]
    data["neighbourhood_group"] = pd.factorize(data["neighbourhood_group"])[0]
    data["neighbourhood"] = pd.factorize(data["neighbourhood"])[0]

    _fig, _ = plt.subplots(figsize=(15, 15))

    sns.heatmap(data[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	
                            "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

    plt.tight_layout()
    plt.show()

    _fig.savefig("correlation_analysis.jpg")

def remove_outliers(data : pd.DataFrame, column : str):
    _q1 = data[column].quantile(0.25)
    _q3 = data[column].quantile(0.75)
    _iqr = _q3 - _q1
    _lower_bound = _q1 - 1.5 * _iqr
    _upper_bound = _q3 + 1.5 * _iqr
    return data[(data[column] >= _lower_bound) & (data[column] <= _upper_bound)]

def scale_data(data : pd.DataFrame):
    _num_variables = ["number_of_reviews", "minimum_nights"]
    scaler = MinMaxScaler()
    scal_features = scaler.fit_transform(data[_num_variables])
    _df_scal = pd.DataFrame(scal_features, index = data.index, columns = _num_variables)
    _df_scal["price"] = data["price"]
    return _df_scal

def data_selection(data_scal : pd.DataFrame):

    _x = data_scal.drop("price", axis = 1)
    _y = data_scal["price"]

    _X_train, _X_test, _y_train, _y_test = train_test_split(_x, _y, test_size = 0.2, random_state = 42)


    _selection_model = SelectKBest(chi2, k = 4)
    _selection_model.fit(_X_train, _y_train)
    _ix = _selection_model.get_support()
    _X_train_sel = pd.DataFrame(_selection_model.transform(_X_train), columns = _X_train.columns.values[_ix])
    _X_test_sel = pd.DataFrame(_selection_model.transform(_X_test), columns = _X_test.columns.values[_ix])

    _X_train_sel["price"] = list(_y_train)
    _X_test_sel["price"] = list(_y_test)

    return _X_train_sel, _X_test_sel




# EDA

df = collect_data(raw_csv = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
save_data(data = df, path = "data/raw/AB_NYC_2019.csv")
print(df.shape)
print(df.info())
print(df.describe())
df = delete_duplicated_values(df)

categorical_analysis(df)
numerical_analysis(df)
correlation_analysis(df)

df = remove_outliers(data = df, column = "price")
df = remove_outliers(data = df, column = "minimum_nights")

df = df[df["price"] > 0]
df = df[df["minimum_nights"] <= 15]

df = delete_irrelevant_information(df)

df_scal = scale_data(df)

train_df, test_df = data_selection(df_scal)

train_df.to_csv('./data/processed/train_data.csv', index=False)
test_df.to_csv('./data/processed/test_data.csv', index=False)







