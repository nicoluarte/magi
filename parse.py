import csv
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path = "/home/nicoluarte/Documents/projects/datarecords"
file_list = glob.glob(path + "/*.csv")
print(file_list)

csv_list = []
final_df = []
for filename in file_list:
    df = pd.read_csv(filename)
    csv_list.append(df)

[x.fillna(method = 'ffill', inplace = True) for x in csv_list]
final_df = pd.concat(csv_list)



df_0 = final_df.dropna()
df_0.replace({'Exercise' : {'HSQ':'SQ', 'BSQ':'SQ', 'SDL':'DL', 'FSQ':'SQ'}},
             inplace = True)
df_0.to_csv(r'/home/nicoluarte/Documents/projects/lift.csv')
data = pd.melt(df_0, id_vars =['Exercise'], value_vars =['MVP(m/s)', 'VMAX(m/s)', 'ROM(cm)', 'POWER(W)'])

g = sns.FacetGrid(data, row="variable",
                  hue="Exercise",
                  sharex=False,
                  sharey=False)
g.map(sns.stripplot, "value")
plt.subplots_adjust(top=0.9, bottom = 0.1)
g.add_legend()
plt.show()

features = ['MVP(m/s)', 'VMAX(m/s)', 'ROM(cm)']
x = df_0.loc[:, features].values
y = df_0.loc[:, ['Exercise']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components,
                            columns=['PCA1', 'PCA2'])

df_0.reset_index(inplace = True)
pca_df = pd.concat([principal_df, df_0['Exercise']], axis = 1)
cln = pd.DataFrame(pca.inverse_transform(principal_components),
                   columns = ['ms', 'pkv', 'rom'])
pca_proj = pd.concat([cln, df_0['Exercise']], axis = 1)

pca_melt = pd.melt(pca_df, id_vars = ['Exercise'],
                   value_vars = ['PCA1', 'PCA2'])


gg = sns.scatterplot(x="index", y="value",
                     data = pca_melt.reset_index(),
                     hue = "Exercise",
                     alpha = 0.3)
plt.show()

new_x = np.array([(0.34, 0.94, 59), (0.3, 0.9, 57), (0.24, 0.46, 29),
                  (0.28, 0.68, 54), (0.3, 0.52, 53)])
pca_new = PCA(n_components=2)
pp = pca_new.fit_transform(new_x)

from sklearn.model_selection import train_test_split
X = pca_df.iloc[:, :-1].values
y = pca_df.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


X = pca_proj.iloc[:, 0:3].values
y = pca_proj.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=14)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

val, idx = min((val, idx) for (idx, val) in enumerate(error))
