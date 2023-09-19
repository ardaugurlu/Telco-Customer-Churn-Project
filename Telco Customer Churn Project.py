
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
 #!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



#İş Problemi:

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

df = pd.read_csv("TelcoChurn/Telco-Customer-Churn.csv")

df.info()

#Görev 1 : Keşifçi Veri Analizi
#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20): #çok önemli bir fonksiyon
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

#manuel olarak
cat_cols = [col for col in df.columns if df[col].dtypes == "O" or df[col].nunique() < 10 ]

for col in df.columns:
    print(col, df[col].nunique())

num_cols = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() > 10]

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

df["TotalCharges"] =  df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["TotalCharges"] = df["TotalCharges"].astype("float")
num_cols = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() > 10]

df.info()
#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.







#Adım 5: Aykırı gözlem var mı inceleyiniz.


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
  print(col ,check_outlier(df, num_cols))


#Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().any()
s
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
#YOK
#Adım 2: Yeni değişkenler oluşturunuz
df.describe().T
df.loc[(df["SeniorCitizen"] == 1) & (df["tenure"] <= 40), "customer_type_age"] = "yenimüşteri_yaşlı"
df.loc[(df["SeniorCitizen"] == 1) & (df["tenure"] > 40), "customer_type_age"] = "eskimüşteri_yaşlı"
df.loc[(df["SeniorCitizen"] == 0) & (df["tenure"] <= 40), "customer_type_age"] = "yenimüşteri_yaşlıdeğil"
df.loc[(df["SeniorCitizen"] == 0) & (df["tenure"] > 40), "customer_type_age"] = "eskimüşteri_yaşlıdeğil"


df["total_reveneu"] = df["MonthlyCharges"] * df["tenure"]

df.loc[(df["DeviceProtection"] == "Yes") & (df["OnlineSecurity"] == "Yes"), "Security_Level"] = "Safety_account"
df.loc[(df["DeviceProtection"] == "Yes") & (df["OnlineSecurity"] == "No"), "Security_Level"] = "Half_safety_account"
df.loc[(df["DeviceProtection"] == "No") & (df["OnlineSecurity"] == "Yes"), "Security_Level"] = "Half_safety_account"
df.loc[(df["DeviceProtection"] == "No") & (df["OnlineSecurity"] == "No"), "Security_Level"] = "Not_Safety"


df.loc[(df["PaymentMethod"] == "Electronic check" ) & (df["PaperlessBilling"] == "Yes"), "Payment_method_andBilling"] = "Electronik_check_Yes_bill"
df.loc[(df["PaymentMethod"] == "Electronic check" ) & (df["PaperlessBilling"] == "No"), "Payment_method_andBilling"] = "Electronik_check_No_bill"
df.loc[(df["PaymentMethod"] == "Mailed check" ) & (df["PaperlessBilling"] == "Yes"), "Payment_method_andBilling"] = "Mailed_check_Yes_bill"
df.loc[(df["PaymentMethod"] == "Mailed check" ) & (df["PaperlessBilling"] == "No"), "Payment_method_andBilling"] = "Mailed_check_No_bill"
df.loc[(df["PaymentMethod"] == "Bank transfer (automatic)" ) & (df["PaperlessBilling"] == "Yes"), "Payment_method_andBilling"] = "Banktransfer(automatic)_Yes_bill"
df.loc[(df["PaymentMethod"] == "Bank transfer (automatic)" ) & (df["PaperlessBilling"] == "No"), "Payment_method_andBilling"] = "Banktransfer(automatic)_No_bill"
df.loc[(df["PaymentMethod"] == "Credit card (automatic)") & (df["PaperlessBilling"] == "Yes"), "Payment_method_andBilling"] = "Creditcard(automatic)_Yes_bill"
df.loc[(df["PaymentMethod"] == "Credit card (automatic)") & (df["PaperlessBilling"] == "No"), "Payment_method_andBilling"] = "Creditcard(automatic)_No_bill"


df["Payment_method_andBilling"].value_counts()



df.loc[(df["SeniorCitizen"] == 1) & (df["gender"] == "Male"), "New_Age_Gender"] = "Senior_Male"
df.loc[(df["SeniorCitizen"] == 1) & (df["gender"] == "Female"), "New_Age_Gender"] = "Senior_Female"
df.loc[(df["SeniorCitizen"] == 0) & (df["gender"] == "Male"), "New_Age_Gender"] = "Not_Senior_Male"
df.loc[(df["SeniorCitizen"] == 0) & (df["gender"] == "Female"), "New_Age_Gender"] = "Not_Senior_Female"


df.set_index("customerID",inplace=True)


def grab_col_names(dataframe, cat_th=10, car_th=20): #çok önemli bir fonksiyon
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


#Adım 3: Encoding işlemlerini gerçekleştiriniz.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]


for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df =one_hot_encoder(df, cat_cols, drop_first=True)

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.



scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])




#Görev 3 : Modelleme
#Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#random forest İLE MODEL
y = df["Churn_1"]
X = df.drop("Churn_1", axis=1)

rf_model = RandomForestClassifier(random_state=17).fit(X,y)
rf_model.get_params()
rf_predict = rf_model.predict(X)
df["predict_churn"] = rf_predict
cv_results = cross_validate(rf_model,
                            X,
                            y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.79
cv_results['test_f1'].mean()
#0.55
cv_results['test_roc_auc'].mean()
#0.82






#KNN İLE MODEL
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
y = df["Churn_1"]
X = df.drop("Churn_1", axis=1)

knn_model = KNeighborsClassifier().fit(X, y) #knn ile modelimizi kurduk
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.76
cv_results['test_f1'].mean()
#0.53
cv_results['test_roc_auc'].mean()
#0.77




#DECİSİON TREE İLE MODEL


import warnings
import joblib
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

y = df["Churn_1"]
X = df.drop("Churn_1", axis=1)
cart_model = DecisionTreeClassifier().fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.723982998903155
cv_results['test_f1'].mean()
# 0.487136016066104
cv_results['test_roc_auc'].mean()
# 0.6512422563219482



#logistic regration ile model

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

y = df["Churn_1"]
X = df.drop("Churn_1", axis=1)
log_model = LogisticRegression(max_iter=1000).fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8032094812568553
cv_results['test_f1'].mean()
# 0.5930962386848326
cv_results['test_roc_auc'].mean()
# 0.8459142476530236
