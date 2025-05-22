# Fake News Classification

This project demonstrates how to classify fake news from real news using machine learning models. It trains and compares four different classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes

## Dataset

The project uses the "Fake and Real News Dataset" from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/

The dataset contains two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

## Requirements

To run this project, you need the following Python packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install them using pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Use

### Step 1: Download the Dataset

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/)
2. Download the dataset (you may need a Kaggle account)
3. Extract the ZIP file
4. Place `Fake.csv` and `True.csv` in a folder named `data` in the project directory

### Step 2: Run the Script

You have two options:

1. Use `fake_news_classification.py` - This will guide you through the process:
```
python fake_news_classification.py
```

2. Use `fake_news_classifier_simple.py` - This assumes you've already downloaded the dataset:
```
python fake_news_classifier_simple.py
```

## What the Script Does

1. **Data Loading and Preprocessing**
   - Loads the fake and real news datasets
   - Adds labels (0 for fake, 1 for real)
   - Combines title and text for better classification

2. **Feature Extraction**
   - Uses TF-IDF Vectorization to convert text into numerical features
   - Splits data into training and testing sets

3. **Model Training**
   - Trains four different classification models on the data

4. **Model Evaluation**
   - Calculates accuracy, precision, recall, and F1 score for each model
   - Generates confusion matrices to visualize model performance
   - Compares training times

5. **Visualization**
   - Creates bar charts comparing model accuracy and training time
   - Plots detailed performance metrics and confusion matrices

## Output

The script generates several visualization files:
- `model_comparison.png`: Bar chart comparing model accuracies
- `detailed_metrics.png`: Comparison of accuracy, precision, recall, and F1 scores
- `training_time.png`: Bar chart comparing model training times
- `confusion_matrices.png`: Confusion matrices for all models

## Expected Results

In general, you can expect:
- Logistic Regression: Good balance of accuracy and speed
- Random Forest: High accuracy but slower training
- Naive Bayes: Fast but slightly lower accuracy
- Decision Tree: Prone to overfitting but reasonably fast

The actual best model may vary depending on the specific data split and random initialization.

## Notes

- The scripts use a fixed random seed (42) for reproducibility
- The models use default parameters for simplicity; performance can be improved with hyperparameter tuning
- Feature extraction limits to 5000 features to balance performance and accuracy
