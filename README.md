# cancer_detection_ANN
Cancer Detection Using Artificial Neural Network

# Breast Cancer Detection using ANN

## Overview
This project aims to detect breast cancer using various features from the breast cancer dataset and an Artificial Neural Network (ANN) for classification. The dataset consists of 569 instances with 33 columns, including diagnostic information and various measurements of cell nuclei.

## Dataset
The dataset used for this project contains the following columns:

- `id`: Unique identifier
- `diagnosis`: The diagnosis of breast tissues (M = malignant, B = benign)
- `radius_mean`: Mean of distances from center to points on the perimeter
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Mean size of the core tumor
- `area_mean`: Mean area of the tumor
- `smoothness_mean`: Mean of local variation in radius lengths
- `compactness_mean`: Mean of perimeter^2 / area - 1.0
- `concavity_mean`: Mean of severity of concave portions of the contour
- `concave points_mean`: Mean for number of concave portions of the contour
- `symmetry_mean`: Mean of symmetry
- `fractal_dimension_mean`: Mean for "coastline approximation" - 1
- `radius_se`: Standard error of distances from center to points on the perimeter
- `texture_se`: Standard error of gray-scale values
- `perimeter_se`: Standard error of size of the core tumor
- `area_se`: Standard error of area of the tumor
- `smoothness_se`: Standard error of local variation in radius lengths
- `compactness_se`: Standard error of perimeter^2 / area - 1.0
- `concavity_se`: Standard error of severity of concave portions of the contour
- `concave points_se`: Standard error for number of concave portions of the contour
- `symmetry_se`: Standard error of symmetry
- `fractal_dimension_se`: Standard error for "coastline approximation" - 1
- `radius_worst`: "Worst" or largest mean value for distances from center to points on the perimeter
- `texture_worst`: "Worst" or largest mean value of gray-scale values
- `perimeter_worst`: "Worst" or largest mean value of size of the core tumor
- `area_worst`: "Worst" or largest mean value of area of the tumor
- `smoothness_worst`: "Worst" or largest mean value of local variation in radius lengths
- `compactness_worst`: "Worst" or largest mean value of perimeter^2 / area - 1.0
- `concavity_worst`: "Worst" or largest mean value of severity of concave portions of the contour
- `concave points_worst`: "Worst" or largest mean value for number of concave portions of the contour
- `symmetry_worst`: "Worst" or largest mean value of symmetry
- `fractal_dimension_worst`: "Worst" or largest mean value for "coastline approximation" - 1
- `Unnamed: 32`: Column with all null values, which was dropped

## Data Preprocessing
1. **Basic Analysis**:
   - Checked the shape, descriptions, and basic information using `df.info()` and `df.describe()`.
   - Verified null values using `df.isnull().sum()`.

2. **Column Dropping**:
   - Dropped the `Unnamed: 32` column as it contained all null values.
   - Dropped the `id` column as it was not relevant for prediction.

3. **Diagnosis Value Counts**:
   - Verified the count of each diagnosis category:
     ```plaintext
     B    357
     M    212
     Name: count, dtype: int64
     ```

4. **Encoding Diagnosis**:
   - Converted the `diagnosis` column to numerical values using the `map` function:
     ```python
     df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
     ```

5. **Correlation Analysis**:
   - Plotted a heatmap to visualize correlations between all columns.
   - Checked the correlation of the `diagnosis` column with other features.

## Model Training
1. **Train-Test Split**:
   - Used `train_test_split` from `sklearn` with stratification on the `diagnosis` column to ensure balanced distribution in the training and testing sets:
     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import MinMaxScaler

     X = df.drop('diagnosis', axis=1)
     y = df['diagnosis']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

     scaler = MinMaxScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     ```

2. **Model Definition**:
   - Defined the ANN model using `Sequential` from `tf.keras.models` and added layers using `Dense` and `Dropout` from `tf.keras.layers`:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense, Dropout

     model = Sequential()
     model.add(Dense(30, activation="relu", input_shape=(X_train.shape[1],)))
     model.add(Dropout(0.2))
     model.add(Dense(15, activation="relu"))
     model.add(Dropout(0.2))
     model.add(Dense(8, activation="relu"))
     model.add(Dropout(0.2))
     model.add(Dense(1, activation="sigmoid"))

     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
     ```

3. **Model Training**:
   - Trained the model with 150 epochs:
     ```python
     model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=1, epochs=150)
     ```

## Model Evaluation
Evaluated the model using accuracy and loss metrics:
- **Accuracy**: 0.9708
- **Loss**: 0.1041

Plotted the training and validation loss and accuracy to visualize model performance over epochs.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/riitk/cancer_detection_ANN.git
   cd cancer_detection_ANN
