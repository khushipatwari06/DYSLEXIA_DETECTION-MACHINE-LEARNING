# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dyt-tablet.csv', delimiter=';')

# Print columns to verify the correct column names
print("Columns in the dataframe:")
print(df.columns)

# Strip whitespace from column names if any
df.columns = df.columns.str.strip()

# Check for the presence of the 'Dyslexia' column
if 'Dyslexia' not in df.columns:
    print("Column 'Dyslexia' not found! Please check the dataset.")
else:
    # Display basic information
    print(df.head())
    print(df.info())

    # Check for missing values and handle them
    print("Missing values in each column:")
    print(df.isnull().sum())

    # Fill missing values with 0 or use another method as appropriate
    df = df.fillna(0)

    # Encode categorical columns
    categorical_columns = ['Gender', 'Nativelang', 'Otherlang']  # Adjust as needed

    # Convert categorical columns to numeric values
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    df['Dyslexia'] = label_encoder.fit_transform(df['Dyslexia'])

    # Split data into features and target variable
    X = df.drop(columns=['Dyslexia'])
    y = df['Dyslexia']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Gradient Boosting Model using XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Make predictions
    y_pred = bst.predict(dtest)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
