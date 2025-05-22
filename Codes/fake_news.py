import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import time

# Single file for fake news classification

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK resources downloaded successfully")

# Load dataset
print("Loading dataset from Datasets folder...")
fake_path = 'Datasets/Fake.csv'
true_path = 'Datasets/True.csv'

# Check if files exist
if not (os.path.exists(fake_path) and os.path.exists(true_path)):
    raise FileNotFoundError(f"Dataset files not found in {os.path.abspath('Datasets')} directory")

# Read CSV files
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Label fake news as 1 and real news as 0python "c:\Users\Zeynel Deniz\Desktop\data2\Codes\fake_news_all_in_one.py"
fake_df['label'] = 1  # 1 for fake news
true_df['label'] = 0  # 0 for real news

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
print(f"Loaded dataset with {len(df)} articles ({len(fake_df)} fake, {len(true_df)} real)")

# Display dataset info
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Sample fake news title: {fake_df['title'].iloc[0]}")
print(f"Sample real news title: {true_df['title'].iloc[0]}")

# Drop irrelevant columns
if 'date' in df.columns:
    df = df.drop(columns=['date'])
if 'subject' in df.columns:
    df = df.drop(columns=['subject'])
print(f"Dropped irrelevant columns. Remaining: {df.columns.tolist()}")

# Text cleaning function
print("Cleaning text data...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple tokenization using split (instead of word_tokenize)
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

# Apply text cleaning
df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_content'] = df['cleaned_title'] + " " + df['cleaned_text']

# Data splitting
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_content'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Vectorization with TF-IDF
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Naive Bayes': MultinomialNB(alpha=0.1)
}

# Train and evaluate models
results = {}
print("\n=== Training and evaluating models ===")

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train_tfidf, y_train)
    
    # Predict
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    training_time = time.time() - start_time
    
    # Save results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'training_time': training_time
    }
    
    # Print results
    print(f"{name} - Accuracy: {accuracy:.4f}, Training time: {training_time:.2f} seconds")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

# Plot results
print("\n=== Visualizing model performance ===")

# Data distribution pie chart
print("Creating label distribution pie chart...")
label_counts = df['label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(
    label_counts,
    labels=[f"{'Fake' if label == 1 else 'Real'} ({count})" for label, count in zip(label_counts.index, label_counts)],
    explode=[0.1, 0],
    colors=['#ff9999','#66b3ff'],
    autopct='%1.1f%%',
    shadow=True,
    startangle=90
)
plt.title("News Label Distribution")
plt.axis('equal')
plt.tight_layout()
plt.savefig('label_distribution.png')
plt.close()

# Accuracy comparison
plt.figure(figsize=(12, 6))
accuracies = [result['accuracy'] for result in results.values()]
model_names = list(results.keys())

bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
plt.ylim(min(accuracies) - 0.05, 1.01)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.title('Model Accuracy Comparison')

# Add accuracy labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_accuracy.png')
plt.close()

# Training time comparison
plt.figure(figsize=(10, 6))
times = [result['training_time'] for result in results.values()]

plt.bar(model_names, times, color='#9b59b6')
plt.ylabel('Training Time (seconds)')
plt.xlabel('Models')
plt.title('Training Time Comparison')
plt.tight_layout()
plt.savefig('training_time.png')
plt.close()

# Precision-Recall comparison
plt.figure(figsize=(14, 7))
width = 0.35
x = np.arange(len(model_names))

precision_fake = [results[model]['report']['1']['precision'] for model in model_names]
recall_fake = [results[model]['report']['1']['precision'] for model in model_names]
precision_real = [results[model]['report']['0']['precision'] for model in model_names]
recall_real = [results[model]['report']['0']['precision'] for model in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Precision comparison
ax1.bar(x - width/2, precision_fake, width, label='Fake News', color='#ff9999')
ax1.bar(x + width/2, precision_real, width, label='Real News', color='#66b3ff')
ax1.set_ylabel('Precision Score')
ax1.set_title('Precision by Model')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.set_ylim(0.8, 1.0)
ax1.legend()

# Recall comparison
ax2.bar(x - width/2, recall_fake, width, label='Fake News', color='#ff9999')
ax2.bar(x + width/2, recall_real, width, label='Real News', color='#66b3ff')
ax2.set_ylabel('Recall Score')
ax2.set_title('Recall by Model')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.set_ylim(0.8, 1.0)
ax2.legend()

plt.tight_layout()
plt.savefig('precision_recall_comparison.png')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Real', 'Fake'],
               yticklabels=['Real', 'Fake'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {name}')
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# ROC Curve for each model
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for name, result in results.items():
    model = result['model']
    
    # Get probabilities for the positive class (fake news)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_tfidf)[:, 1]
    else:
        # For models that don't have predict_proba (like some SVM configurations)
        y_score = model.decision_function(X_test_tfidf) if hasattr(model, "decision_function") else model.predict(X_test_tfidf)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Word frequency visualization for fake vs. real news
from collections import Counter
import itertools

# Get most common words in fake and real news
fake_words = ' '.join(df[df['label'] == 1]['cleaned_content']).split()
real_words = ' '.join(df[df['label'] == 0]['cleaned_content']).split()

fake_word_counts = Counter(fake_words)
real_word_counts = Counter(real_words)

# Get the 15 most common words for each category
most_common_fake = dict(fake_word_counts.most_common(15))
most_common_real = dict(real_word_counts.most_common(15))

# Plot word frequencies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Fake news words
ax1.bar(most_common_fake.keys(), most_common_fake.values(), color='#ff9999')
ax1.set_title('Most Common Words in Fake News')
ax1.set_ylabel('Frequency')
ax1.tick_params(axis='x', rotation=45)

# Real news words
ax2.bar(most_common_real.keys(), most_common_real.values(), color='#66b3ff')
ax2.set_title('Most Common Words in Real News')
ax2.set_ylabel('Frequency')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('word_frequency.png')
plt.close()

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

# Print full summary
print("\nModel performance summary:")
summary = []
for name, result in results.items():
    report = result['report']
    summary.append({
        'Model': name,
        'Accuracy': result['accuracy'],
        'Precision (Fake)': report['1']['precision'],
        'Recall (Fake)': report['1']['recall'],
        'F1 (Fake)': report['1']['f1-score'],
        'Precision (Real)': report['0']['precision'],
        'Recall (Real)': report['0']['recall'],
        'F1 (Real)': report['0']['f1-score'],
        'Training Time': result['training_time']
    })

summary_df = pd.DataFrame(summary)
print(summary_df[['Model', 'Accuracy', 'F1 (Fake)', 'F1 (Real)', 'Training Time']])

print("\n=== Fake News Classification completed ===")
print("Generated visualization files:")
print("- model_accuracy.png: Bar chart showing accuracy for each model")
print("- training_time.png: Bar chart showing training time for each model")
print("- confusion_matrices.png: Confusion matrices for all models")
print("- label_distribution.png: Pie chart showing distribution of fake and real news")
print("- precision_recall_comparison.png: Comparison of precision and recall for each model")
print("- roc_curve.png: ROC curves for all models")
print("- word_frequency.png: Most common words in fake and real news")
