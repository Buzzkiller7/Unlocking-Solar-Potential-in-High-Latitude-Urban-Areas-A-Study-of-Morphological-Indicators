import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
import shap
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import xgboost as xgb

def data_preprocessing(df):
    x = df.drop(columns=['SP', 'OID_', 'Shape_Length', 'Shape_Area', 'Minority_use', 'Majority_use', 'N', 'SP_Classification'])
    y = df['SP_Classification']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.176, random_state=42) #0.176*0.85=0.15

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler

def load_best_params(model_name):
    filename = os.path.join(directory, model_name + '_best_params.pkl')
    if os.path.exists(filename):
        print(f"Loading best parameters for {model_name}...")
        best_params = joblib.load(filename)
    else:
        best_params = None
    return best_params

def train_model(model, param_grid, x_train, y_train, model_name):
    filename = os.path.join(directory, model_name + '.pkl')
    if os.path.exists(filename):
        print(f"Loading {model_name} model...")
        model = joblib.load(filename)
    else:
        print(f"Training {model_name} model...")
        best_params = load_best_params(model_name)
        if best_params is None:
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            best_params = grid_search.best_params_
            joblib.dump(best_params, filename + '_best_params.pkl')
        model.set_params(**best_params)
        model.fit(x_train, y_train)
        joblib.dump(model, filename)
    return model

def evaluate_model(model, x, y, data_name, model_name):
    y_pred = model.predict(x)
    accuracy = model.score(x, y)
    print(f"Accuracy of {model_name} on {data_name}: {accuracy}")

    # 计算F1分数
    f1 = f1_score(y, y_pred, average='macro')
    print(f"F1 Score of {model_name} on {data_name}: {f1}")

    # 计算ROC AUC得分
    y_pred_proba = model.predict_proba(x)  # 获取每个类的概率
    auc_score = roc_auc_score(y, y_pred_proba, multi_class='ovr')
    print(f"ROC AUC Score of {model_name} on {data_name}: {auc_score}")

df = pd.read_csv('E:/Study/python/RoadClip/Road_Clip_Domestic_Classified2.csv')
x_train, x_val, x_test, y_train, y_val, y_test, scaler = data_preprocessing(df)
directory = 'E:/Study/python/RoadClip/B_models_01'
if not os.path.exists(directory):
    os.makedirs(directory)

def create_dnn_model():
    model = Sequential()
    model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax')) # 3 classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

models = {
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Artificial Neural Network": MLPClassifier(max_iter=1000), # Increase maximum iterations for ANN
    "XGBoost": XGBClassifier(objective="multi:softmax", num_class=3, eval_metric='mlogloss'),
    "Support Vector Machine": SVC(probability=True),
    "Deep Neural Network": KerasClassifier(build_fn=create_dnn_model, epochs=50, batch_size=32, verbose=0)
}

param_grid = {
    "Linear Discriminant Analysis": {},
    "Random Forest": {"n_estimators": [10, 50, 100, 200],
                      "max_depth": [None, 10, 20, 30, 40],
                      "max_features": list(range(1, 19))},
    "Naive Bayes": {},
    "K-Nearest Neighbors": {"n_neighbors": list(range(1, 102))},
    "Artificial Neural Network": {"hidden_layer_sizes": [(i,) for i in range(2, 21)],
                                  "alpha": [i/20 for i in range(0, 11)]},
    "XGBoost": {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3]
    },
    "Support Vector Machine": {"C": [0.01, 0.1, 1, 10, 100, 1000],
                               "kernel": ['linear', 'poly', 'rbf', 'sigmoid']},
    "Deep Neural Network": {"epochs": [50, 100, 150, 200], "batch_size": [32, 64]}
}

models_param_grid = [(models[name], param_grid[name], name) for name in models]

for model, param_grid, model_name in tqdm(models_param_grid):
    try:
        model = train_model(model, param_grid, x_train, y_train, model_name)
        evaluate_model(model, x_train, y_train, 'train set', model_name)
        evaluate_model(model, x_val, y_val, 'validation set', model_name)
        evaluate_model(model, x_test, y_test, 'test set', model_name)
    except Exception as e:
        print(f'Error occurred while processing {model_name}: {e}')
        continue

# 提取特征值
filename = os.path.join(directory, 'XGBoost' + '.pkl')
xgb_model = joblib.load(filename)
feature_names = ['BD', 'PR', 'BH', 'BF', 'BA', 'BFA', 'BS', 'BO', 'BP', 'BV', 'std_BH', 'std_BF', 'std_BA', 'std_BFA', 'std_BS', 'std_BO', 'std_BP', 'std_BV']

data = pd.DataFrame(x_train)
data.columns = feature_names

# data = data.sample(frac=0.1)  # 使用10%的数据

explainer = shap.TreeExplainer(xgb_model, data[feature_names])  # 为XGBoost模型创建解释器

shap_values = explainer.shap_values(data[feature_names])
y_base = explainer.expected_value
print(y_base)

data['pred'] = xgb_model.predict(data[feature_names])
print(data['pred'].mean())

shap.initjs()
for i in range(len(y_base)):  # 为每个类别分别绘制 SHAP 值
    shap.force_plot(explainer.expected_value[i], shap_values[i], feature_names=feature_names)
    shap.summary_plot(shap_values[i], data[feature_names])
    shap.summary_plot(shap_values[i], data[feature_names], plot_type="bar")

num_features = len(feature_names)

# 计算每个类别的SHAP值的平均
mean_shap_values = [np.abs(shap_values[i]).mean(axis=0) for i in range(3)]

# 找到每个类别中最重要的5个特征
top_features = [np.argsort(mean_shap_values[i])[-6:] for i in range(3)]

for i in range(3):
    for j, feature in enumerate(top_features[i]):
        print(f'Class {i}, Feature: {feature_names[int(feature)]}')

# 计算整体的SHAP值
shap_values_sum = np.sum(shap_values, axis=0)

shap.initjs()

# 绘制整体的SHAP值
shap.summary_plot(shap_values_sum, data[feature_names])
shap.summary_plot(shap_values_sum, data[feature_names], plot_type="bar")

num_features = len(feature_names)

# 计算整体的SHAP值的平均
mean_shap_values = np.abs(shap_values_sum).mean(axis=0)

# 找到最重要的5个特征
top_features = np.argsort(mean_shap_values)[-6:]

for j, feature in enumerate(top_features):
    print(f'Feature: {feature_names[int(feature)]}')

# 绘制一个包含所有类别的SHAP总结图
shap.summary_plot(shap_values, data[feature_names])

num_features = len(feature_names)