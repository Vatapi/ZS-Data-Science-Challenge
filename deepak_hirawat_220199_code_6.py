#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix ,accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
seaborn.set(style='ticks')
dataset = pd.read_csv("data.csv")


# In[2]:


dataset.columns


# In[3]:


dataset.location_x.mode()


# In[4]:


dataset["location_y"].mode()


# In[5]:


dataset.location_x.fillna(0.0, inplace = True)
dataset["location_y"].fillna(0.0 , inplace =True)


# In[6]:


dataset.isna().sum()


# In[7]:


dataset["shot_id_number"].interpolate(method="linear" , ax=1 , inplace=True)
count = 0
for i in tqdm(range(len(dataset))):
    if dataset["shot_id_number"][i] == i+1:
        count += 1
count


# In[8]:


dataset["game_season"].value_counts()


# In[9]:


dataset["game_season"].fillna(method="ffill" , inplace=True)
dataset["knockout_match"].fillna(method ="bfill" , inplace=True)


# In[10]:


dataset["team_id"].unique()


# In[11]:


dataset["team_name"].unique()


# In[12]:


def loc(col , is_string):
    bool_series = dataset[col].isna()
    testdata = dataset[bool_series]
    false_series = dataset[col].notnull()
    training_data = dataset[false_series]
    
    df = pandas.DataFrame({
        'loc_x': training_data['location_x'],
        'loc_y': training_data['location_y'],
        'Col': training_data[col]
    })
    df1 = pandas.DataFrame({
        'loc_x': testdata['location_x'],
        'loc_y': testdata['location_y'],
        'Col': testdata[col]
    })
    fg = seaborn.FacetGrid(data=df, hue='Col', aspect=2)
    fg.map(plt.scatter, 'loc_x', 'loc_y').add_legend()
    
    X_train = df.iloc[: , 0:2].values
    y_train = df.iloc[: , 2].values
    X_test = df1.iloc[: , 0:2].values
    y_test = df1.iloc[: , 2].values
    
    if is_string == 1:
        label_enc = LabelEncoder()
        y_train = label_enc.fit_transform(y_train.astype(str))
    
    knn = KNeighborsClassifier(n_neighbors = 17)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    if is_string==1:
        y_pred = label_enc.inverse_transform(y_pred)
    dataset[col] = dataset[col].replace(np.nan , 'missing')
    j = 0
    for i in tqdm(range(len(dataset))):
        if dataset[col][i] == 'missing':
            dataset[col][i] = y_pred[j]
            j+=1
    dataset[col] = dataset[col].replace('missing' , np.nan)
    return y_pred


# In[13]:


values = loc("distance_of_shot" , 0)
values


# In[14]:


val = loc("range_of_shot" , 1)
val


# In[15]:


valu = loc("area_of_shot" , 1)
valu


# In[16]:


value = loc("shot_basics" , 1)
value


# In[17]:


def same_columns(x,y,power):
    count = 0
    dataset[x] = dataset[x].replace(np.nan , 'x')
    dataset[y] = dataset[y].replace(np.nan , 'x')
    for i in tqdm(range(len(dataset))):
        if dataset[x][i] == 'x' and dataset[y][i] != 'x':
            if power == 1 and dataset[y][i]<=7:
                dataset[x][i] = dataset[y][i]
            elif power!=1:
                dataset[x][i] = dataset[y][i]
            else:
                pass
        elif dataset[x][i] == 'x' and dataset[y][i] == 'x':
            count += 1
        else:
            pass
    dataset[x] = dataset[x].replace('x' , np.nan)
    dataset[y] = dataset[y].replace('x' , np.nan)
    return count


# In[18]:


a = same_columns("power_of_shot" , "power_of_shot.1",1)
a


# In[19]:


b = same_columns("remaining_min" , "remaining_min.1",0)
b


# In[20]:


c = same_columns("remaining_sec" , "remaining_sec.1",0)
c


# In[21]:


c = same_columns("type_of_shot" , "type_of_combined_shot" , 0)
c


# In[22]:


dataset["remaining_time"] = dataset["remaining_min"] + (dataset["remaining_sec"]/60)
dataset.drop("remaining_min" , axis =1, inplace =True)
dataset.drop("remaining_sec" , axis =1, inplace =True)
dataset.drop("team_id" , axis=1,inplace=True)
dataset.drop("team_name" , axis=1 ,inplace=True)
dataset.drop("match_event_id" , axis =1, inplace=True)
dataset.drop("knockout_match.1" , axis =1,inplace =True)
dataset.drop("distance_of_shot.1" ,axis=1,inplace=True)
dataset.drop("type_of_combined_shot" ,axis=1,inplace=True)
dataset.drop(dataset.columns[0] , axis=1,inplace=True)
dataset.drop("power_of_shot.1" , axis=1,inplace=True)
dataset.drop("remaining_min.1" ,axis=1, inplace=True)
dataset.drop("remaining_sec.1" ,axis=1, inplace=True)


# In[23]:


def one_to_one(col):
    dic = {}
    for i in tqdm(range(len(dataset))):
        if i!=22902:
            if (dataset[col][i] is not np.nan):
                dic[dataset["match_id"][i]] = dataset[col][i]
    for i in tqdm(range(len(dataset))):
        if i!= 22902:
            if dataset[col][i] is np.nan:
                x = dataset["match_id"][i]
                dataset[col][i] = dic[x]
    temp = dataset.groupby("match_id")[col].transform(lambda x: x.nunique()==1)
    return temp.all()


# In[24]:


x = one_to_one("home/away")
x


# In[25]:


y = one_to_one("lat/lng")
y


# In[26]:


z = one_to_one("date_of_game")
z


# In[27]:


dataset[dataset["match_id"] == 29600031]


# In[28]:


dataset.drop(22902 , axis=0 , inplace =True)


# In[29]:


z = one_to_one("date_of_game")
z


# In[30]:


bool_series = dataset["is_goal"].isna()
testdata = dataset[bool_series]
false_series = dataset["is_goal"].notnull()
training_data = dataset[false_series]


# In[31]:


training_data.isna().sum()


# In[32]:


testdata.isna().sum()


# In[33]:


training_data.dropna(inplace=True)


# In[34]:


training_data.isna().sum()


# In[35]:


med =testdata["power_of_shot"].median()


# In[36]:


testdata["power_of_shot"].fillna(med , inplace=True)


# In[37]:


med = testdata["remaining_time"].median()
med


# In[38]:


testdata["remaining_time"].fillna(med ,inplace=True)


# In[39]:


testdata.isna().sum()


# In[40]:


training_data.isna().sum()


# In[41]:


len(training_data)


# In[42]:


len(testdata)


# In[43]:


training_data.to_csv("cleaned_training_data2.csv" , index=False)


# In[44]:


testdata.to_csv("cleaned_test_data2.csv" , index=False)


# In[45]:


df_train = pd.read_csv("cleaned_training_data2.csv")
df_test = pd.read_csv("cleaned_test_data2.csv")
df_test.columns


# In[46]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train["game_season"] = le.fit_transform(df_train.iloc[:,4].values)
df_test["game_season"] = le.transform(df_test.iloc[:,4].values)

one_hot = pd.get_dummies(df_train["area_of_shot"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("area_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_train["range_of_shot"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("range_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_train["shot_basics"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("shot_basics" , axis=1 , inplace=True)

one_hot = pd.get_dummies(df_train["home/away"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("home/away" , axis=1 , inplace=True)

one_hot = pd.get_dummies(df_train["type_of_shot"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("type_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_train["lat/lng"] , drop_first=True)
df_train = df_train.join(one_hot)
df_train.drop("lat/lng" , axis=1,inplace=True)
df_train.drop("date_of_game" , axis=1 , inplace=True)
df_train.drop("shot_id_number" ,axis=1 ,inplace=True)
#df_train.drop("match_id" , axis=1 ,inplace=True)


# In[47]:


one_hot = pd.get_dummies(df_test["area_of_shot"] , drop_first=True)
df_test = df_test.join(one_hot)
df_test.drop("area_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_test["range_of_shot"] , drop_first=True)
df_test = df_test.join(one_hot)
df_test.drop("range_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_test["shot_basics"] , drop_first=True)
df_test = df_test.join(one_hot)
df_test.drop("shot_basics" , axis=1 , inplace=True)

one_hot = pd.get_dummies(df_test["home/away"] , drop_first=True)
df_test = df_test.join(one_hot , lsuffix="left" , rsuffix="right")
df_test.drop("home/away" , axis=1 , inplace=True)

one_hot = pd.get_dummies(df_test["type_of_shot"] , drop_first=True)
df_test = df_test.join(one_hot)
df_test.drop("type_of_shot" , axis=1, inplace=True)

one_hot = pd.get_dummies(df_test["lat/lng"] , drop_first=True)
df_test = df_test.join(one_hot)
df_test.drop("lat/lng" , axis=1,inplace=True)
df_test.drop("date_of_game" , axis=1 , inplace=True)
df_test.drop("shot_id_number" ,axis=1 ,inplace=True)
#df_test.drop("match_id" , axis=1 ,inplace=True)


# In[48]:


x = df_train.iloc[: , df_train.columns != "is_goal"].values
y = df_train.iloc[: , df_train.columns == "is_goal"].values


# In[49]:


X = df_test.iloc[: , df_train.columns != "is_goal"].values
Y = df_test.iloc[: , df_train.columns == "is_goal"].values

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[50]:


from xgboost import XGBClassifier
model = XGBClassifier(max_depth=2 ,learning_rate=0.05)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[51]:


X = sc.transform(X)
y_pred = model.predict_proba(X)
temp = pd.read_csv("cleaned_test_data2.csv")
temp["shot_id_number"] = temp["shot_id_number"].astype('int64')
result = pd.DataFrame(data={"shot_id_number" : temp["shot_id_number"], "is_goal" : y_pred[:, 1]})
result


# In[52]:


result.to_csv("deepak_hirawat_220199_6.csv", index=False)


# In[53]:


df_test = df_test.iloc[: , df_test.columns != "is_goal"]
importances = pd.DataFrame({"columns":df_test.columns , "importances":model.feature_importances_})
importances.sort_values("importances" ,ascending=False)


# In[ ]:




