const express = require('express');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());
// Example GET route: /get-letter?filename=example.jpg
const info1 = `

`;
const info2 = `

`;
const info3 = `

`;
const info4 = `

`;
const info5 = `

`;
const info6 = `

`;
const info7 = `

`;
const info8 = `

`;
const info9 = `

`;
const info10 = `

`;

const ml1= `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/student/Downloads/cust_data.csv')

df.info()
df.head()

num_col= 'age'
mean = df[num_col].mean()
median = df[num_col].median()
mode = df[num_col].mode()[0]
std_dev = df[num_col].std()
variance = df[num_col].var()
data_range = df[num_col].max() - df[num_col].min()
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {data_range}")

plt.figure(figsize=(4, 3))
sns.histplot(df[num_col], bins=10, kde=True, color='blue')
plt.title(f'Histogram of {num_col}')
plt.xlabel(num_col)
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(4,3))
sns.boxplot(x=df[num_col], color='green')
plt.title(f'Boxplot of {num_col}')
plt.show()

Q1 = df[num_col].quantile(0.25)
Q3 = df[num_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[num_col] < lower_bound) | (df[num_col] > upper_bound)][num_col]
print(f"Number of outliers detected: {len(outliers)}")

cat_col= 'gender'
counts = df[cat_col].value_counts()
print(counts)

plt.figure(figsize=(8,4))
sns.barplot(x=counts.index, y=counts.values)
plt.title(f"Bar chart of {cat_col}")
plt.xlabel(cat_col)
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,5))
plt.pie(counts, labels = counts.index, autopct='%1.1f%%')
plt.title(f"Pie chart of {cat_col}")
plt.show()
`;
const ml2= `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
x_col = 'sepal_length'
y_col = 'petal_length'
data.describe()

correlation = df[[x_col,y_col]].corr('pearson')
print("Pearson Correlation Coefficient:\n", correlation)

covariance = df[[x_col,y_col]].cov()
print("Covariance Matrix:\n", covariance)

plt.figure(figsize=(8, 5))
plt.scatter(df[x_col], df[y_col])
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f"Scatter Plot of {x_col} vs {y_col}")
plt.show()

data_co = df.iloc[:, :-1]
covariance_matrix = data_co.cov()
correlation_matrix = data_co.corr()
print("Covariance Matrix:\n", covariance_matrix)
print("\n Correlation Matrix:\n", correlation_matrix)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
`;
const ml3= `
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

df_pca = pd.DataFrame(X_pca, columns=['PC1','PC2'])
df_pca['Species'] = y

plt.figure(figsize=(6,4))
colors = ['brown','hotpink','purple']
for i, color in zip(np.unique(y), colors): plt.scatter(df_pca.loc[df_pca['Species']==i,'PC1'],df_pca.loc[df_pca['Species']==i,'PC2'],c=color,label=iris.target_names[i])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.show()
`;
const ml4= `
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

def cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False):
   results = {}
   for k in k_values:
       if weighted:
           knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
       else:
           knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
       knn.fit(X_train, y_train)
       y_pred = knn.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1-score for multi-class
       results[k] = {'accuracy': accuracy, 'f1_score': f1}
   return results

k_values = [1, 3, 5]

print("Regular k-NN Results:")
regular_knn = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False)
for k, metrics in regular_knn.items():
   print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")

print("\nWeighted k-NN Results:")
weighted_knn = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=True)
for k, metrics in weighted_knn.items():
   print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")

print("\nComparison of Regular k-NN and Weighted k-NN:")
for k in k_values:
   regular_acc = regular_knn[k]['accuracy']
   weighted_acc = weighted_knn[k]['accuracy']
   print(f"k={k}: Regular k-NN Accuracy={regular_acc:.4f}, Weighted k-NN Accuracy={weighted_acc:.4f}")
`;
const ml6= `
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y = y + 10 * np.sin(X[:, 0] * 2)

plt.scatter(X, y, color='blue', label='Data Points')
plt.title("Synthetic Dataset")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

def locally_weighted_regression(X, y, query_point, tau=0.1):
    weights = np.exp(-np.sum((X - query_point) ** 2, axis=1) / (2 * tau ** 2))
    X_bias = np.c_[np.ones(X.shape[0]), X]
    W = np.diag(weights)
    theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
    query_point_bias = np.array([1, query_point[0]])
    y_pred = query_point_bias @ theta
    return y_pred

def predict_lwr(X_train, y_train, X_test, tau=0.1):
    y_pred = np.zeros(X_test.shape[0])
    for i, query_point in enumerate(X_test):
        y_pred[i] = locally_weighted_regression(X_train, y_train, query_point, tau)
    return y_pred

X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
tau = 0.1
y_pred = predict_lwr(X, y, X_test, tau)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_pred, color='red', label='LWR Fit')
plt.title(f"Locally Weighted Regression (tau={tau})")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

mse = mean_squared_error(y, predict_lwr(X, y, X, tau))
print(f"Mean Squared Error (MSE) on Training Data: {mse:.4f}")
`;

const ml7 = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston_df = pd.read_csv("boston_housing_data.csv")

print("Linear Regression on Boston Housing Dataset")

X = boston_df[['RM']]
y = boston_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

auto_df = pd.read_csv("auto-mpg.csv")
print("Polynomial Regression on Auto MPG Dataset")

auto_df['horsepower'] = auto_df['horsepower'].replace('?', np.nan).astype(float)
auto_df.dropna(inplace=True)

X = auto_df[['horsepower']]
y = auto_df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

PR_model = LinearRegression()
PR_model.fit(X_train_poly, y_train)
y_pred = PR_model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.scatter(X_test, y_test, color='purple', label='Actual')
sorted_indices = X_test.squeeze().argsort()
plt.plot(X_test.iloc[sorted_indices], y_pred[sorted_indices], color='red', label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG (Miles Per Gallon)')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.legend()
plt.show()
`;

const ml8 = `
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = sns.load_dataset('titanic')

print(data.head())
print(data.info())

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

data = data[features + ['survived']].dropna()

data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['embarked'] = data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = data[features]
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree for Titanic Dataset")
plt.show()

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
`;

const ml9 = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes classifier: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show()
`;

const ml10 = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
ari_score = adjusted_rand_score(y_true, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Adjusted Rand Index: {ari_score:.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans, palette="coolwarm", s=60)

plt.title('K-Means Clustering Result (PCA-reduced data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_true, palette="Set2", s=60)
plt.title('True Labels (PCA-reduced data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.legend(title="Actual Class")

plt.grid(True)

plt.show()
`;
app.get('/info1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info1);   // return the Python code
});
app.get('/info2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info2);   // return the Python code
});
app.get('/info3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info3);   // return the Python code
});
app.get('/info4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info4);   // return the Python code
});
app.get('/info5', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info5);   // return the Python code
});
app.get('/info6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info6);   // return the Python code
});
app.get('/info7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info7);   // return the Python code
});
app.get('/info8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info8);   // return the Python code
});
app.get('/info9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info9);   // return the Python code
});
app.get('/info10', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info10);   // return the Python code
});
app.get('/ml1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml1);   // return the Python code
});
app.get('/ml2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml2);   // return the Python code
});
app.get('/ml3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml3);   // return the Python code
});
app.get('/ml4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml4);
});
app.get('/ml6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml6);   // return the Python code
});
app.get('/ml7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml7);   // return the Python code
});
app.get('/ml8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml8);   // return the Python code
});
app.get('/ml9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml9);   // return the Python code
});
app.get('/ml10', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml10);   // return the Python code
});
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
