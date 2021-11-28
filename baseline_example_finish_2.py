#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Загрузка библиотек

import pandas as pd
import numpy as np
import re


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
# In[ ]:

# Создание списка и словаря для 
# числового отоброжения диагноза
# список также использутеся для исключения значений в val,
# которых нет в train

diag_list = [
             'A', 'B', 'C', 'D', 'E',
             'G', 'H', 'I', 'J', 'K', 
             'L','M', 'N', 'O', 'Q', 
             'R', 'S', 'T', 'V', 'Z'
            ]


diag_dict ={}

for i, v in enumerate(diag_list):
    diag_dict[v] = i
    


# In[ ]:

# Функция для формирования датафреймов val и train
# Получаем данные температуры, массы тела, роста, диагноза
# Удаляем пропущенные значения

def spliting_data(df):
    
    df['Temp'] = np.nan
    df['Weight'] = np.nan
    df['Height'] = np.nan
    df['Diag_1'] = ''
    
    for index, row in df.iterrows():
        if isinstance(row[1], str):
            if ('Температура тела : ' in row[1]):
                result = re.search('Температура тела : ([0-9]{2}[,][0-9]?) С', row[1])
                if result:
                    df.iat[index,3] = float(str(result.group(1)).replace(',', '.'))

            if ('Веc, кг' in row[1]):
                result = re.search('Веc, кг : ([0-9]{1,3})', row[1])
                if result:
                    df.iat[index,4] = int(result.group(1))

            if ('Рост, см :' in row[1]):
                result = re.search('Рост, см : ([0-9]{1,3})', row[1])
                if result:
                    df.iat[index,5] = int(result.group(1))

            if ('ДИАГНОЗ' in row[1]):
                result = re.search(': ([A-Z][0-9]{2}.?[0-9]?)', row[1])
                if result:
                    df.iat[index,6] = str(result.group(1))[0]

                if ('МКБ10' in row[1]):
                    result = re.search(': ([A-Z][0-9]{2}.?[0-9]?)', row[1])
                    if result:
                        df.iat[index,6] = str(result.group(1))[0]
    df = df.drop(['id','text'], axis = 1)
    df = df.query("Diag_1 != ''")
    
    return df


# In[ ]:


# Функция, которая удаляет ошибочно введенные значения val и train
# в данных температуры, массы тела, роста, диагноза
# Рассчитываем новую метрику BMI - индекс массы тела
# Здесь нас ожидал сюрприз, данные BMI выходили за пределы 10 и 60,
# что указывает на ошибки заполения документации. 

def cleaning(df):
    
    df['BMI'] = df['Weight']/(df['Height']/100)**2
    df = df.loc[df['Temp'] < 42] 
    df = df.loc[df['Temp'] > 33]
    df = df[df['Height'] > 53]
    df = df[df['Height'] <= 220]
    df = df[df['Weight'] > 2.5]
    df = df[df['Weight'] <= 180]    
    df = df[df['BMI'] > 10]
    df = df[df['BMI'] <= 60] 
    
    
    df = df[df['Diag_1'].isin(diag_list)] 
    df = df.reset_index(drop = True)
    
    return df
    


# In[ ]:

# Функция, которая проверяет наличие диагноза по словарю
# Важно, чтобы модель обучалсь и предсказывала на однотипных данных.

def add_cat(row):
    return diag_dict[row]


# In[ ]:

# Базовая функция, которая собирает все функции воедино
# Загрузка, регулярные выражения, очистка и удаления ненужных для ML
# колонок
def biseline_data(name):
    df = pd.read_csv(name, on_bad_lines = 'warn', engine = 'python')
    df = spliting_data(df)
    df = cleaning(df)
    df['Diag_cat'] = df['Diag_1'].apply(add_cat)
    df = df.drop('Diag_1', axis = 1)
    return df
    


# In[ ]:
# Загрузка и получение инофрмация о датафреймах для ML

df_train = biseline_data('train.csv')

df_train.info()


# In[ ]:


df_val = biseline_data('val.csv')

df_val.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_val.describe()


# In[ ]:


# Подготовка данных к обучению и прогнозу
# Разделение целевой метрики и признаков для обучения




target = df_train['label']
features = df_train.drop('label', axis=1)
val_target =  df_val['label']
val_features = df_val.drop('label', axis=1)

# Балансировка датасета для обучения
# Уменьшаем количество выживших пациентов

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled

# Балансировка
features_downsampled, target_downsampled = downsample(features, target, 0.03)

# Оценка размера балансированных датасетов

print(features_downsampled.shape)
print(target_downsampled.shape)


# Снижение размерности значений датасета

sc = StandardScaler()
features_downsampled = sc.fit_transform(features_downsampled)
val_features = sc.transform(val_features)


# Обучение и получение прогноза
clf=RandomForestClassifier(n_estimators=100)

clf.fit(features_downsampled,target_downsampled)

y_pred=clf.predict(val_features)

# Оценка качества модели
print("Метрики:",metrics.accuracy_score(val_target, y_pred))
print("F1",f1_score(val_target, y_pred, average=None))

# По результатам мы видим, что получилась модель с низкой прогностической
# способностью.

