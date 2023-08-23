
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder

# Read the CSV file
data = pd.read_csv('D:\Vscode\codes\PYTHON\dataanalytics_lab\hello.csv')

# Select numeric attributes
numeric_data = data[['numeric_attr1', 'numeric_attr2']]

# Select nominal attributes
nominal_data = data[['nominal_attr1', 'nominal_attr2']]

# Preprocess nominal attributes using one-hot encoding
encoder = OneHotEncoder()
nominal_data_encoded = encoder.fit_transform(nominal_data).toarray()

# Calculate dissimilarity matrix for numeric attributes
dissimilarity_matrix_numeric = pairwise_distances(numeric_data, metric='euclidean')

# Calculate similarity matrix for numeric attributes
similarity_matrix_numeric = 1 / (1 + dissimilarity_matrix_numeric)

# Calculate dissimilarity matrix for nominal attributes
dissimilarity_matrix_nominal = pairwise_distances(nominal_data_encoded, metric='jaccard')

# Calculate similarity matrix for nominal attributes
similarity_matrix_nominal = 1 - dissimilarity_matrix_nominal

# Print dissimilarity matrix for numeric attributes
print("Dissimilarity between numeric attributes:")
print(pd.DataFrame(dissimilarity_matrix_numeric))

# Print similarity matrix for numeric attributes
print("\nSimilarity between numeric attributes:")
print(pd.DataFrame(similarity_matrix_numeric))

# Print dissimilarity matrix for nominal attributes
print("\nDissimilarity between nominal attributes:")
print(pd.DataFrame(dissimilarity_matrix_nominal))

# Print similarity matrix for nominal attributes
print("\nSimilarity between nominal attributes:")
print(pd.DataFrame(similarity_matrix_nominal))
