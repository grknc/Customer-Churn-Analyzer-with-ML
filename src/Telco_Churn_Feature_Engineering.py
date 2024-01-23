"""
Author : Mustafa Gürkan Çanakçi
LinkedIn : https://www.linkedin.com/in/mgurkanc/
"""

# Project Name : Telco Churn Feature Engineering

# Business Problem
# In this project, we will develop a machine learning model that
# can predict customers who will leave the company.
# Before modelling , we will make the exploratory data analysis and feature engineering for its dataset.

# Content of Variables:
# CustomerID
# Gender
# SeniorCitizen
# Partner
# Dependents
# tenure
# PhoneService
# MultipleLines
# InternetService
# OnlineSecurity
# OnlineBackup
# DeviceProtection
# TechSupport
# StreamingTV
# StreamingMovies
# Contract
# PaperlessBiling
# PaymentMethod
# MonthlyCharges
# TotalCharges
# Churn - Diabetic ( 1 or 0 )



########################################################################################
#                           1.EXPLORATORY DATA ANALYSIS                                #
########################################################################################

# * 1.1.Importing necessary libraries*

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


######################
# * 1.2.Read the dataset*
######################
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

# * Checking the data*
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)


####################################################################################
# * 1.3.Define a Function to grab the Numerical and Categorical variables of its dataset*
####################################################################################

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


cat_cols
# ['gender',
#  'Partner',
#  'Dependents',
#  'PhoneService',
#  'MultipleLines',
#  'InternetService',
#  'OnlineSecurity',
#  'OnlineBackup',
#  'DeviceProtection',
#  'TechSupport',
#  'StreamingTV',
#  'StreamingMovies',
#  'Contract',
#  'PaperlessBilling',
#  'PaymentMethod',
#  'Churn',
#  'SeniorCitizen']

num_cols
# Out[10]: ['tenure', 'MonthlyCharges', 'TotalCharges']


#####################################
# * 1.5.Target Variable Analysis
#####################################

df["Churn"].value_counts()
# Out[148]:
# No     5174
# Yes    1869
# Name: Churn, dtype: int64

df["Churn"] = df["Churn"].map({'No':0,'Yes':1})

df["Churn"].value_counts()
# Out[13]:
# 0    5174
# 1    1869
# Name: Churn, dtype: int64

# Target Summary with Categorical variables

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"CHURN_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("###################################")

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

df.head()

# Target Summary with Numerical variables
def target_summary_with_num(dataframe,target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n")
    print("###################################")

for col in num_cols:
    target_summary_with_num(df,"Churn",col)


def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################################################")

for col in cat_cols:
    cat_summary(df,col)


###############################################
# * 1.4.Outliers Analysis
###############################################

# Define a Function about outlier threshold for data columns
def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Define a Function about checking outlier for data columns
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Define a Function about replace with threshold for data columns
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


################################
# * 1.5.The Missing Values Analysis
################################

# Define a Function about missing values for dataset columns
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


#########################
# * 1.6.Correlation Analysis
#########################

corr_matrix = df[num_cols].corr()
corr_matrix


########################################################################################
#                           2.FEATURE ENGINEERING                                      #
########################################################################################


###############################################
# * 2.1.Processing for Missing Values and Outliers
###############################################
df.isnull().sum()
na_cols = missing_values_table(df, True)

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

df.isnull().sum()


###############################################
# * 2.2.Creating New Feature Interactions
###############################################

df.head()

def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################################################")

for col in cat_cols:
    cat_summary(df,col)

df["gender"] = df["gender"].map({'Male':0,'Female':1})
df.head()

# # Create a Senior/Young Man & Old/Young Woman Categorical variable
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"]== 1)), 'SENIOR/YOUNG_GENDER'] ="senior_male"
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"]== 0)), 'SENIOR/YOUNG_GENDER'] ="young_male"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"]== 1)), 'SENIOR/YOUNG_GENDER'] ="senior_female"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"]== 0)), 'SENIOR/YOUNG_GENDER'] ="young_female"

df.groupby("SENIOR/YOUNG_GENDER").agg({"Churn": ["mean","count"]})


# Telefon hizmeti kullanan/kullanmayan kadın/erkek müşteriler
df.loc[((df['gender'] == 0) & (df["PhoneService"]== "Yes")), 'PHONE_SER_GENDER'] ="phone_ser_male"
df.loc[((df['gender'] == 0) & (df["PhoneService"]== "No")), 'PHONE_SER_GENDER'] ="no_phone_ser_male"
df.loc[((df['gender'] == 1) & (df["PhoneService"]== "Yes")), 'PHONE_SER_GENDER'] ="phone_service_female"
df.loc[((df['gender'] == 1) & (df["PhoneService"]== "No")), 'PHONE_SER_GENDER'] ="no_phone_ser_female"

df.groupby("PHONE_SER_GENDER").agg({"Churn": ["mean","count"]})

# Baklamakla yükümlü olan kadın/erkek müşteriler
df.loc[((df['gender'] == 0) & (df["Dependents"]== "Yes")), 'DEPEND_GENDER'] ="dependent_male"
df.loc[((df['gender'] == 0) & (df["Dependents"]== "No")), 'DEPEND_GENDER'] ="undependent_male"
df.loc[((df['gender'] == 1) & (df["Dependents"]== "Yes")), 'DEPEND_GENDER'] ="dependent_female"
df.loc[((df['gender'] == 1) & (df["Dependents"]== "No")), 'DEPEND_GENDER'] ="undependent_female"

df.groupby("DEPEND_GENDER").agg({"Churn": ["mean","count"]})

# Bir/Birden Fazla ya da hiç hattı olmayan erkek/kadın müşterilerin kategorisi
df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "Yes")), 'PHONE_LINE_GENDER'] ="multiple_lines__male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "No")), 'PHONE_LINE_GENDER'] ="single_line__male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "No phone service")), 'PHONE_LINE_GENDER'] ="no_line__male"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "Yes")), 'PHONE_LINE_GENDER'] ="multiple_lines__female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "No")), 'PHONE_LINE_GENDER'] ="single_line__female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "No phone service")), 'PHONE_LINE_GENDER'] ="no_line__female"

df.groupby("PHONE_LINE_GENDER").agg({"Churn": ["mean","count"]})

# Ödeme yöntemi farklı olan kadın/erkek müşterilerin kategorisi
df.loc[((df['gender'] == 0) & (df["PaymentMethod"]== "Electronic check")), 'GENDER_PAYMENT'] ="male_electronic_check_pay"
df.loc[((df['gender'] == 0) & (df["PaymentMethod"]== "Mailed check")), 'GENDER_PAYMENT'] ="male_mailed_check_pay"
df.loc[((df['gender'] == 0) & (df["PaymentMethod"]== "Bank transfer (automatic)")), 'GENDER_PAYMENT'] ="male_bank_transfer_pay"
df.loc[((df['gender'] == 0) & (df["PaymentMethod"]== "Credit card (automatic)")), 'GENDER_PAYMENT'] ="male_credit_card_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"]== "Electronic check")), 'GENDER_PAYMENT'] ="female_electronic_check_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"]== "Mailed check")), 'GENDER_PAYMENT'] ="female_mailed_check_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"]== "Bank transfer (automatic)")), 'GENDER_PAYMENT'] ="female_bank_transfer_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"]== "Credit card (automatic)")), 'GENDER_PAYMENT'] ="female_credit_card_pay"


df.groupby("GENDER_PAYMENT").agg({"Churn": ["mean","count"]})

# Kadın/Erkek Müşterilerin Sözleşme Süresi
df.loc[((df['gender'] == 0) & (df["Contract"]== "Month-to-month")), 'GENDER_CONTRACT'] ="male_monthly_contract"
df.loc[((df['gender'] == 0) & (df["Contract"]== "One year")), 'GENDER_CONTRACT'] ="male_one_year_contact"
df.loc[((df['gender'] == 0) & (df["Contract"]== "Two year")), 'GENDER_CONTRACT'] ="male_two_year_contact"
df.loc[((df['gender'] == 1) & (df["Contract"]== "Month-to-month")), 'GENDER_CONTRACT'] ="female_monthly_contract"
df.loc[((df['gender'] == 1) & (df["Contract"]== "One year")), 'GENDER_CONTRACT'] ="female_one_year_contact"
df.loc[((df['gender'] == 1) & (df["Contract"]== "Two year")), 'GENDER_CONTRACT'] ="female_two_year_contact"

df.groupby("GENDER_CONTRACT").agg({"Churn": ["mean","count"]})

# Yaşlı/Genç Kadın/Erkek Müşterilerin Internet Servis Sağlayıcısının Kategorisi
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_male") & (df["InternetService"]== "Fiber optic")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_male_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_male") & (df["InternetService"]== "DSL")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_male_dsl_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_male") & (df["InternetService"]== "No")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_male_no_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_male") & (df["InternetService"]== "Fiber optic")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_male_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_male") & (df["InternetService"]== "DSL")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_male_dsl_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_male") & (df["InternetService"]== "No")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_male_no_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_female") & (df["InternetService"]== "Fiber optic")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_female_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_female") & (df["InternetService"]== "DSL")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_female_dsl_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "senior_female") & (df["InternetService"]== "No")), 'SEN_YNG_GENDER_INTERNET_SER'] ="senior_female_no_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_female") & (df["InternetService"]== "Fiber optic")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_female_fiber_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_female") & (df["InternetService"]== "DSL")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_female_dsl_internet"
df.loc[((df['SENIOR/YOUNG_GENDER'] == "young_female") & (df["InternetService"]== "No")), 'SEN_YNG_GENDER_INTERNET_SER'] ="young_female_no_internet"

df.groupby("SEN_YNG_GENDER_INTERNET_SER").agg({"Churn": ["mean","count"]})


# Kadın/Erkek İnternet Servis,Çevrimiçi Güvenlik ve Yedeği olan Kadın/Erkek Müşterilerin Kategorisi

df.loc[((df['gender'] == 0) & (df["InternetService"]== "Fiber optic") & ((df["OnlineSecurity"]== "Yes"))), 'INT_SEC_SERV_GENDER'] ="male_fiber_int_security"
df.loc[((df['gender'] == 0) & (df["InternetService"]== "Fiber optic") & ((df["OnlineSecurity"]== "No"))), 'INT_SEC_SERV_GENDER'] ="male_fiber_int_no_security"
df.loc[((df['gender'] == 0) & (df["InternetService"]== "DSL") & ((df["OnlineSecurity"]== "Yes"))), 'INT_SEC_SERV_GENDER'] ="male_dsl_int_security"
df.loc[((df['gender'] == 0) & (df["InternetService"]== "DSL") & ((df["OnlineSecurity"]== "No"))), 'INT_SEC_SERV_GENDER'] ="male_dsl_int_no_security"
df.loc[((df['gender'] == 1) & (df["InternetService"]== "Fiber optic") & ((df["OnlineSecurity"]== "Yes"))), 'INT_SEC_SERV_GENDER'] ="female_fiber_int_security"
df.loc[((df['gender'] == 1) & (df["InternetService"]== "Fiber optic") & ((df["OnlineSecurity"]== "No"))), 'INT_SEC_SERV_GENDER'] ="female_fiber_int_no_security"
df.loc[((df['gender'] == 1) & (df["InternetService"]== "DSL") & ((df["OnlineSecurity"]== "Yes"))), 'INT_SEC_SERV_GENDER'] ="female_dsl_int_security"
df.loc[((df['gender'] == 1) & (df["InternetService"]== "DSL") & ((df["OnlineSecurity"]== "No"))), 'INT_SEC_SERV_GENDER'] ="female_dsl_int_no_security"

df.groupby("INT_SEC_SERV_GENDER").agg({"Churn": ["mean","count"]})

###############################################
# * 2.3.Processing Encoding and One-Hot Encoding
###############################################
le = LabelEncoder()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)

df.info()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 30 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

###############################################
# * 2.4.Standardization for numerical variables
###############################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()


###############################################
# * 2.5.Create Modelling
###############################################

y = df["Churn"]
X = df.drop(["customerID","Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
