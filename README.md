# Human Activity Recognition (HAR) Classification

This project builds a machine learning model to classify human activities from smartphone sensor data. Using accelerometer and gyroscope readings, the model predicts whether someone is walking, walking upstairs, walking downstairs, sitting, standing, or laying down.

## The Approach

I started by exploring the dataset to understand what we're working with - 561 features extracted from sensor signals across 30 different subjects performing 6 activities. The data was already pretty clean (no missing values), but I needed to handle the preprocessing carefully.

The key challenge here was making sure the model generalizes to new people, not just new data from the same people. I used standard scaling (z-score normalization) on all features, which is crucial when you have sensor data with different units and scales. The label encoder handles the categorical activity labels.

I went with a Random Forest classifier because it's robust, handles the high-dimensional feature space well, and gives us feature importance scores which are really useful for understanding what the model is actually using to make decisions.

## Hyperparameter Tuning

I spent some time tuning the hyperparameters and managed to bump the validation accuracy from 97.76% to 97.94%. The small improvement margin makes sense as the model was already performing well, so large gains couldent really happen at that point.

What I found interesting is that multiple parameter combinations yielded the exact same performance.
 
This makes sense because:
- The model is probably near its ceiling - 97.94% might be close to the best possible performance for this dataset
- Random Forest is pretty robust and less sensitive to hyperparameter changes than some other algorithms
- There's always some variance in cross-validation, so small differences can round to the same percentage

The final model uses 200 estimators with a max depth of 20, which seems to be a good balance between performance and complexity.

## The Reality Check

Here's where it gets interesting. When I train on the full training set and test on the completely separate test set, the performance drops to 92.67%. That's a noticeable drop, but it's actually revealing something important about the problem.

The test set has no overlapping subjects with the training set - completely new people. Even though I removed the subject column from the features, the model can still learn patterns that are specific to individual subjects (like how they walk, their gait, etc.) through the sensor data itself. When those same subjects show up in the validation split, the model does great. But when it encounters completely new people in the test set, it has to rely purely on generalizable activity patterns, which is harder.

This is a good reminder that train/validation/test splits need to respect the structure of your data. For this kind of problem, you really want to split by subjects, not randomly, to get a realistic sense of how the model will perform in the real world.

## What I Learned

The biggest takeaway for me was understanding the difference between validation performance and true generalization. A 97.94% validation accuracy looks great, but the 92.67% test accuracy on unseen subjects is the number that actually matters for deployment. That's still pretty solid performance, but it's a more honest assessment.

I also got better at thinking through feature engineering and preprocessing. Standard scaling was essential here, and the feature importance plots helped me understand which sensor measurements actually matter for distinguishing activities.

## Project Structure

- `src/train.py` - Main training script that loads data, preprocesses, trains the model, and evaluates
- `src/evaluate.py` - Visualization functions for confusion matrices, feature importance, and activity distributions
- `src/utils.py` - Utility functions
- `models/` - Saved model, scaler, and encoder
- `results/` - Generated plots and visualizations
- `data/` - Training and test datasets

## Running the Project

Just run `python3 train.py` from `cd src`. It'll train the model, evaluate it, save everything to the models directory, and generate all the visualizations.
