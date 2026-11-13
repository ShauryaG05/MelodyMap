import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
CSV_PATH = "./features.csv"


try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")


    # X = features_df.drop('label', axis=1)
    X = features_df.drop(columns=["filename", "label"])
    
    y = features_df['label'].astype(str)

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)


    print("\nVerifying the separation:")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    print("\nFirst 5 rows of features (X):")
    print(X[:5])

    print("\nFirst 5 values of target (y):")
    print(y[:5])



    print("\n--- Label Encoding Step ---")
    
    # Check the data type of the target variable y
    # np.issubdtype checks if a dtype is a subtype of another (e.g., is int64 a subtype of integer?)
    if np.issubdtype(y.dtype, np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        # This block would run if your labels were strings (e.g., 'blues', 'classical')
        print("Labels are not numerical. Applying LabelEncoder.")
        



        label_encoder = LabelEncoder()

        # # 2. Fit the encoder to the labels and transform them in one step
        # # The fit_transform method learns the mapping and applies it.
        y = label_encoder.fit_transform(y)

        # # You can inspect the learned classes and their mapping
        print("\nLabelEncoder learned the following classes:")
        print(label_encoder.classes_)
        #
        # # Later, you could use `label_encoder.inverse_transform([0, 1])` to get ['blues', 'classical']
    
    print("\nVerification of target variable 'y':")
    print(f"Data type of y: {y.dtype}")
# Model Traing


    print("\n--- Splitting Data into Training and Testing Sets ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create an instance of StandardScaler and fit it on the training data
    print("\n--- Scaling Features ---")
    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # --- Verification Step ---
    print("StandardScaler has been fitted to the training data.")
    # We can inspect the learned means to verify. This will be an array of 28 means.
    print(f"Learned means (μ) for the first 5 features: {scaler.mean_[:5]}")
    print(f"Learned standard deviations (σ) for the first 5 features: {scaler.scale_[:5]}")


    # Standard Scaler




    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    print("\nVerification of scaled data:")
    

    print(f"Mean of first 5 features in X_train_scaled: {X_train_scaled[:, :5].mean(axis=0)}")
    
    # The standard deviation of each feature in the scaled training data should be very close to 1.
    print(f"Standard deviation of first 5 features in X_train_scaled: {X_train_scaled[:, :5].std(axis=0)}")

    # Note: The mean and std of X_test_scaled will NOT be exactly 0 and 1,
    # because it was scaled using the parameters from X_train. This is correct!
    print(f"\nMean of first 5 features in X_test_scaled: {X_test_scaled[:, :5].mean(axis=0)}")



    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000)

    log_reg.fit(X_train_scaled, y_train)

    print("Logistic Regression model trained successfully!")
    

    print(f"Model learned the following classes: {log_reg.classes_}")




    print("\n--- Training Support Vector Machine (SVM) Model ---")



    svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)

    svm_model.fit(X_train_scaled, y_train)


    print("Support Vector Machine model trained successfully!")
    print(f"Model learned the following classes: {svm_model.classes_}")

    print("\n--- Training Random Forest Classifier Model ---")







    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf_model.fit(X_train_scaled, y_train)

    print("Random Forest Classifier model trained successfully!")
    print(f"Model learned the following classes: {rf_model.classes_}")




    print("\n--- Evaluating Models on the Test Set ---")

    # 1. Logistic Regression

    y_pred_log_reg = log_reg.predict(X_test_scaled)
    # Compare the predicted labels with the true labels to calculate accuracy
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    # 2. Support Vector Machine
    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")

    # 3. Random Forest Classifier
    # An alternative and more concise way to get accuracy is the .score() method,
    # which performs prediction and scoring in one step.
    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")


    print("\n--- Saving Models and Scaler to Disk ---")

    # The first argument is the Python object to save.
    # The second argument is the desired filename.
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")


except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")