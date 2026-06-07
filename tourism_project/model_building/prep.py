# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
datasetPath = "hf://datasets/PaviRaju/tourism-wellness-package-purchase/tourism.csv"
tourismDataSet = pd.read_csv(datasetPath)
print("Dataset loaded successfully.")

# Print the shape of the dataset
print(f"Shape of the cleaned dataset: {tourismDataSet.shape}")

# Check for duplicates
print("Number of duplicate rows:", tourismDataSet.duplicated().sum())

# Remove duplicates
tourismDataSet = tourismDataSet.drop_duplicates()

# Check for null values
print("Null values per column:\n", tourismDataSet.isnull().sum())

# Handle missing values: Fill numerical NaNs with the mean and categorical NaNs with the mode.
for column in tourismDataSet.columns:
    if tourismDataSet[column].isnull().any():
        if pd.api.types.is_numeric_dtype(tourismDataSet[column]):
            # Fill numeric NaNs with mean
            tourismDataSet[column] = tourismDataSet[column].fillna(tourismDataSet[column].mean())
        else:
            # Fill categorical NaNs with mode, fallback to 'Unknown'
            mode_val = tourismDataSet[column].mode()[0] if not tourismDataSet[column].mode().empty else 'Unknown'
            tourismDataSet[column] = tourismDataSet[column].fillna(mode_val)

print(f"Shape of the cleaned dataset: {tourismDataSet.shape}")
print("Missing values after cleaning:\n",tourismDataSet.isnull().sum())

# Treat the Gender column
# Replace "Fe Male" with "Female" in the Gender column
tourismDataSet['Gender'] = tourismDataSet['Gender'].replace("Fe Male", "Female")
print(tourismDataSet['Gender'].unique())

# Treat the Marital Status Columns
# Replace "Unmarried" with "Single" in MaritalStatus column
tourismDataSet['MaritalStatus'] = tourismDataSet['MaritalStatus'].replace("Unmarried", "Single")
print(tourismDataSet['MaritalStatus'].unique())

# Drop the unique column 
tourismDataSet = tourismDataSet.drop(columns=['CustomerID'], errors='ignore')

# Define the target variable for the classification task
target = 'ProdTaken'

# Define predictor matrix (X) using selected numeric and categorical features
X = tourismDataSet.drop(columns=[target])

# Define target variable
y = tourismDataSet[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="PaviRaju/tourism-wellness-package-purchase",
        repo_type="dataset",
    )
