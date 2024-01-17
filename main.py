import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Create a random DataFrame
np.random.seed(42)
data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randint(1, 10, 100),
    'Feature3': np.random.choice(['A', 'B', 'C'], 100),
    'Feature4': np.random.choice([np.nan, 5, 10], 100)  # A column with missing values
}

df = pd.DataFrame(data)
print("Created DataFrame:")
print(df.head())

# Introduce missing values
df.loc[df.sample(frac=0.2).index, 'Feature4'] = np.nan

# Convert the categorical column to numerical
label_encoder = LabelEncoder()
df['Feature3'] = label_encoder.fit_transform(df['Feature3'])

# Detect missing values
missing_value_count = df.isnull().sum()
print("\nMissing Value Count:")
print(missing_value_count)

# Apply KNN Imputer to fill missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Visualization
plt.figure(figsize=(10, 6))

# Original data set
plt.subplot(2, 1, 1)
plt.title('Original Data Set')
plt.bar(df.columns, df.isnull().sum(), color='red')
plt.ylabel('Number of Missing Values')

# Imputed data set
plt.subplot(2, 1, 2)
plt.title('Imputed Data Set')
plt.bar(df_imputed.columns, df_imputed.isnull().sum(), color='green')
plt.ylabel('Number of Missing Values')

plt.tight_layout()
plt.show()