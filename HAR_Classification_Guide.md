# Human Activity Recognition (HAR) Classification Project Guide

## Project Overview
This guide will walk you through building a supervised learning model to classify 6 human activities using sensor data from accelerometers and gyroscopes.

**Target Activities:**
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

---

## Step 1: Loading and Exploring the Dataset

### Why This Step Matters
Before building any model, you need to understand your data. This helps you:
- Identify data quality issues (missing values, outliers)
- Understand feature distributions
- Check class balance
- Plan appropriate preprocessing steps

### Concepts to Understand

**DataFrame Structure:**
- Your dataset has 563 columns: 561 features + 'subject' + 'Activity'
- Each row represents a time window of sensor measurements
- Features are extracted statistics (mean, std, energy, etc.) from raw accelerometer/gyroscope signals

**Key Exploratory Questions:**
1. How many samples do we have?   2
2. Are there missing values? no
3. How are activities distributed? (class balance)
4. How many unique subjects?  30
5. What's the range of feature values? 

### Your Task: Write Code to Load and Explore

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the training and test datasets
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# TODO: Print basic information about the datasets
# - Shape (rows, columns)
print(train_df.shape)
# - Column names
# - Data types
# - First few rows

# TODO: Check for missing values
# - Use .isnull().sum() to find missing values per column
# - Use .isnull().sum().sum() to find total missing values

# TODO: Explore the target variable (Activity)
# - Use .value_counts() to see activity distribution
# - Create a bar plot showing activity counts

# TODO: Explore the subject column
# - How many unique subjects?
# - How are subjects distributed across activities?

# TODO: Get basic statistics for features
# - Use .describe() to see mean, std, min, max for numeric columns
```

**Questions to Answer:**
1. How many training samples? How many test samples?
2. Are there any missing values? If yes, how should we handle them?
3. Is the dataset balanced across activities?
4. What's the range of feature values? (This will inform scaling decisions)

---

## Step 2: Separating Features and Target Labels

### Why This Step Matters
Machine learning models need:
- **X (features)**: The input variables the model learns from
- **y (target)**: The output variable we want to predict

We need to separate these clearly before preprocessing.

### Concepts to Understand

**Feature Selection:**
- Exclude metadata columns ('subject', 'Activity') from features
- Keep all 561 feature columns for now (we can do feature selection later)
- The 'subject' column might be useful for cross-validation strategies, but typically not used as a feature

**Label Encoding:**
- Machine learning models work with numbers, not strings
- We need to convert activity names (strings) to numeric labels
- Use LabelEncoder from sklearn.preprocessing

### Your Task: Write Code to Separate Features and Labels

```python
from sklearn.preprocessing import LabelEncoder

# TODO: Separate features from target for training set
# - Create X_train: all columns except 'subject' and 'Activity'
# - Create y_train: just the 'Activity' column

# TODO: Do the same for test set
# - Create X_test and y_test

# TODO: Encode activity labels to numbers
# - Create a LabelEncoder instance
# - Fit it on y_train (this learns the mapping)
# - Transform both y_train and y_test
# - Print the mapping (which number corresponds to which activity)

# TODO: Check the shapes
# - Print shapes of X_train, y_train, X_test, y_test
# - Verify they make sense
```

**Questions to Answer:**
1. How many features are we using?
2. How many classes (activities) do we have?
3. What's the mapping between activity names and encoded numbers?

---

## Step 3: Data Preprocessing

### Why This Step Matters
Real-world data is rarely ready for machine learning. Preprocessing ensures:
- **Scaling**: Features on different scales don't dominate the model
- **Handling Missing Values**: Models can't work with NaN values
- **Data Quality**: Clean data leads to better models

### Concepts to Understand

**Feature Scaling:**
- Different features have different scales (e.g., -1 to 1 vs 0 to 1000)
- Algorithms like SVM, Neural Networks, and distance-based methods are sensitive to scale
- **StandardScaler**: Transforms features to have mean=0 and std=1 (z-score normalization)
- **MinMaxScaler**: Transforms features to range [0, 1]
- **Why Random Forest doesn't need scaling**: Tree-based methods split on thresholds, so scale doesn't matter

**Train-Test Split for Scaling:**
- **CRITICAL**: Fit the scaler ONLY on training data
- Then transform both training and test data
- This prevents data leakage (using test set information during training)

**Missing Values:**
- Check if your dataset has missing values
- If yes, options: drop rows, impute (fill with mean/median), or use algorithms that handle NaN

### Your Task: Write Code for Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# TODO: Check for missing values again
# - Are there any NaN values in X_train or X_test?

# TODO: Create a StandardScaler
# - Instantiate StandardScaler()
# - Fit it on X_train (learns mean and std from training data)
# - Transform X_train and X_test

# TODO: Convert back to DataFrame (optional, but helpful for exploration)
# - Create new DataFrames with scaled features
# - Preserve column names for interpretability

# TODO: Verify scaling worked
# - Check mean and std of scaled features (should be ~0 and ~1)
```

**Questions to Answer:**
1. Why do we fit the scaler only on training data?
2. What would happen if we scaled the entire dataset together?
3. Should we scale for Random Forest? Why or why not?

---

## Step 4: Model Selection and Training

### Why This Step Matters
Different algorithms have different strengths:
- **Random Forest**: Robust, interpretable, handles non-linearity, no scaling needed
- **XGBoost**: Often best accuracy, handles complex patterns, requires tuning
- **MLP (Neural Network)**: Can learn complex patterns, but needs scaling and more tuning

### Concepts to Understand

**Train-Validation Split:**
- We need a validation set to tune hyperparameters
- Use `train_test_split` to split training data into train + validation
- Typical split: 80% train, 20% validation

**Random Forest:**
- Ensemble of decision trees
- Each tree votes, majority wins
- Hyperparameters:
  - `n_estimators`: Number of trees (more = better but slower)
  - `max_depth`: Maximum depth of trees (controls overfitting)
  - `min_samples_split`: Minimum samples to split a node
  - `random_state`: For reproducibility

**Overfitting:**
- Model memorizes training data but fails on new data
- Signs: High training accuracy, low validation accuracy
- Solutions: Reduce model complexity, add regularization, get more data

### Your Task: Write Code for Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# TODO: Split training data into train and validation sets
# - Use train_test_split with test_size=0.2, random_state=42
# - This gives us: X_train_split, X_val, y_train_split, y_val

# TODO: Create and train a Random Forest model
# - Start with: n_estimators=100, random_state=42
# - Fit on X_train_split, y_train_split

# TODO: Make predictions
# - Predict on training set (to check for overfitting)
# - Predict on validation set (to evaluate performance)

# TODO: Evaluate the model
# - Calculate accuracy on training and validation sets
# - Print classification report for validation set
# - What's the gap between train and validation accuracy?
```

**Questions to Answer:**
1. What's your training accuracy vs validation accuracy?
2. Is there overfitting? (Large gap = overfitting)
3. Which classes are hardest to predict? (Check classification report)

---

## Step 5: Hyperparameter Tuning

### Why This Step Matters
Default hyperparameters are rarely optimal. Tuning improves:
- Model performance
- Generalization (reducing overfitting)
- Efficiency (faster training/inference)

### Concepts to Understand

**Grid Search vs Random Search:**
- **GridSearchCV**: Tries all combinations of specified hyperparameters (exhaustive)
- **RandomizedSearchCV**: Tries random combinations (faster, often finds good solutions)
- **Cross-Validation**: Splits data into k folds, trains on k-1, validates on 1 (repeated k times)

**Key Hyperparameters for Random Forest:**
- `n_estimators`: More trees = better but slower (try 50, 100, 200)
- `max_depth`: Deeper = more complex (try 10, 20, None for unlimited)
- `min_samples_split`: Higher = simpler model (try 2, 5, 10)

### Your Task: Write Code for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# TODO: Define parameter grid
# - Create a dictionary with hyperparameters to try
# - Example: {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}

# TODO: Create GridSearchCV or RandomizedSearchCV
# - Use RandomForestClassifier as estimator
# - Specify cv=5 for 5-fold cross-validation
# - Use scoring='accuracy'
# - Set n_jobs=-1 to use all CPU cores

# TODO: Fit the search
# - Fit on X_train_split, y_train_split
# - This will try all combinations and find the best

# TODO: Get best parameters and best score
# - Print best_params_ and best_score_

# TODO: Train final model with best parameters
# - Create new RandomForestClassifier with best parameters
# - Fit on full training set (X_train, y_train)
```

**Questions to Answer:**
1. What are the best hyperparameters?
2. How much did performance improve?
3. How long did tuning take? (Consider if RandomizedSearchCV would be faster)

---

## Step 6: Model Evaluation on Test Set

### Why This Step Matters
The test set is your final, unbiased evaluation:
- It simulates real-world performance
- Never use it for training or tuning (that's cheating!)
- It tells you how well your model will generalize

### Concepts to Understand

**Confusion Matrix:**
- Shows actual vs predicted for each class
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Helps identify which classes are confused

**Classification Metrics:**
- **Accuracy**: Overall correctness (can be misleading if classes are imbalanced)
- **Precision**: Of predicted positives, how many were actually positive
- **Recall**: Of actual positives, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall

**Per-Class Performance:**
- Some activities might be harder to distinguish (e.g., SITTING vs STANDING)
- Understanding per-class metrics helps identify model weaknesses

### Your Task: Write Code for Test Set Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# TODO: Make predictions on test set
# - Use your trained model to predict X_test

# TODO: Calculate test accuracy
# - Compare predictions with y_test

# TODO: Create confusion matrix
# - Use confusion_matrix(y_test, y_pred)
# - Create a heatmap visualization with seaborn
# - Add labels for activities

# TODO: Print detailed classification report
# - Use classification_report(y_test, y_pred)
# - Shows precision, recall, F1-score for each class

# TODO: Analyze results
# - Which activities are most confused?
# - What's the overall test accuracy?
```

**Questions to Answer:**
1. What's your final test accuracy?
2. Which two activities are most often confused?
3. Why might certain activities be harder to distinguish?

---

## Step 7: Feature Importance and Interpretability

### Why This Step Matters
Understanding which features matter:
- Builds trust in the model
- Reveals insights about the problem
- Helps with feature engineering
- Great for presentations/resumes!

### Concepts to Understand

**Feature Importance (Random Forest):**
- Each tree splits on features that best separate classes
- Features used in important splits get higher importance scores
- Sum of all importances = 1.0
- Higher importance = more predictive power

**Visualization:**
- Bar plots of top N features
- Helps identify which sensor signals are most informative

### Your Task: Write Code for Feature Importance

```python
# TODO: Get feature importances from trained model
# - Access .feature_importances_ attribute

# TODO: Create a DataFrame with feature names and importances
# - Sort by importance (descending)

# TODO: Visualize top 20 most important features
# - Create a horizontal bar plot
# - Label axes clearly

# TODO: Analyze results
# - Which types of features are most important?
# - Are time-domain or frequency-domain features more useful?
# - Do accelerometer or gyroscope features dominate?
```

**Questions to Answer:**
1. What are the top 5 most important features?
2. Do the important features make intuitive sense?
3. Could you reduce the number of features without losing much accuracy?

---

## Step 8: Visualizations and Results Summary

### Why This Step Matters
Clear visualizations:
- Communicate results effectively
- Make your project resume-friendly
- Help stakeholders understand the model
- Reveal patterns in the data

### Concepts to Understand

**Effective Visualizations:**
- **Confusion Matrix Heatmap**: Shows classification errors clearly
- **Feature Importance Plot**: Shows what the model relies on
- **Activity Distribution**: Shows class balance
- **Prediction Probabilities**: Shows model confidence

### Your Task: Create Comprehensive Visualizations

```python
# TODO: Create a figure with multiple subplots
# - Activity distribution (bar plot)
# - Confusion matrix (heatmap)
# - Top feature importances (bar plot)
# - Maybe: Prediction probabilities distribution

# TODO: Add titles, labels, and formatting
# - Make it publication-ready
# - Use clear, descriptive titles

# TODO: Save the figure
# - Use plt.savefig() with high DPI
```

**Questions to Answer:**
1. How would you present these results to a non-technical audience?
2. What insights can you draw from the visualizations?

---

## Step 9: (Optional) Anomaly Detection Extension

### Why This Step Matters
Real-world applications need to handle:
- Unusual activity patterns
- Sensor malfunctions
- Activities not seen during training
- Low-confidence predictions

### Concepts to Understand

**Anomaly Detection Using Model Confidence:**
- Prediction probabilities indicate confidence
- Low max probability = uncertain prediction = potential anomaly
- Can flag predictions where max probability < threshold (e.g., 0.7)

**Applications:**
- Quality control: Flag unusual sensor readings
- Safety: Detect unexpected movements
- Data collection: Identify samples that need human review

### Your Task: Implement Anomaly Detection

```python
# TODO: Get prediction probabilities
# - Use .predict_proba() instead of .predict()

# TODO: Calculate confidence scores
# - Max probability for each prediction = confidence

# TODO: Set a confidence threshold
# - Try 0.7 or 0.8
# - Flag predictions below threshold as anomalies

# TODO: Analyze anomalies
# - How many anomalies in test set?
# - What activities have low confidence?
# - Visualize confidence distribution

# TODO: (Advanced) Investigate low-confidence samples
# - Look at feature values for anomalies
# - Are they outliers in feature space?
```

**Questions to Answer:**
1. What percentage of predictions are flagged as anomalies?
2. Do anomalies correspond to specific activities?
3. How would you use this in a real application?

---

## Next Steps and Best Practices

### Code Organization
- Create functions for each major step (load_data, preprocess, train_model, etc.)
- Use a main() function to orchestrate everything
- Add docstrings to explain what each function does

### Model Improvements to Try
1. **Feature Engineering**: Create new features from existing ones
2. **Feature Selection**: Remove low-importance features to reduce complexity
3. **Try Other Models**: XGBoost, SVM, Neural Networks
4. **Ensemble Methods**: Combine multiple models

### Documentation
- Add comments explaining your choices
- Document hyperparameters and their rationale
- Keep a log of experiments and results

### Version Control
- Use Git to track changes
- Commit after each major milestone
- Write clear commit messages

---

## Summary Checklist

- [ ] Loaded and explored the dataset
- [ ] Separated features and target labels
- [ ] Preprocessed data (scaling, handling missing values)
- [ ] Split data into train/validation sets
- [ ] Trained initial model
- [ ] Tuned hyperparameters
- [ ] Evaluated on test set
- [ ] Analyzed feature importance
- [ ] Created visualizations
- [ ] (Optional) Implemented anomaly detection

---

## Key Takeaways

1. **Always explore your data first** - Understanding data prevents mistakes
2. **Never use test set for training** - Prevents overfitting and false confidence
3. **Scale features appropriately** - Some algorithms need it, others don't
4. **Tune hyperparameters systematically** - Grid search or random search
5. **Evaluate thoroughly** - Accuracy alone isn't enough, check per-class performance
6. **Interpret your model** - Feature importance builds trust and insights
7. **Visualize everything** - Good plots communicate better than numbers

Good luck with your project! 🚀

