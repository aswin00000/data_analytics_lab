import pandas as pd
from scipy.stats import pearsonr

# Read the CSV file
data = pd.read_csv('D:\Vscode\codes\PYTHON\dataanalytics_lab\correlation_input.csv')

# Select the numeric and nominal attributes
numeric_attribute = data['numeric_attr']
nominal_attribute = data['nominal_attr']

# Convert nominal attribute to numerical labels
nominal_labels = nominal_attribute.astype('category').cat.codes

# Calculate the Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(numeric_attribute, nominal_labels)

print("Pearson Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)

# Interpret the correlation
if abs(correlation_coefficient) >= 0.7:
    print("Strong correlation")
elif abs(correlation_coefficient) >= 0.3:
    print("Moderate correlation")
else:
    print("Weak or no correlation")
