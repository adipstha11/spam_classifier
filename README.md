# SMS Spam Classifier

A beginner-friendly machine learning project that classifies SMS messages as `ham` (not spam) or `spam` using Python and scikit-learn.

## Project Overview

This project uses the SMS Spam Collection dataset and compares multiple text classification models using the same preprocessing and train/test split.

### What this project does

- Loads and cleans SMS data
- Preprocesses text (lowercasing, punctuation removal, and whitespace cleanup)
- Converts text to numeric features using TF-IDF
- Trains and evaluates multiple classifiers
- Compares models using:
  - Accuracy
  - Classification report
  - Confusion matrix
  - Spam-class precision, recall, and F1
  - False positives (ham predicted as spam)

## Models Compared

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest
- K-Nearest Neighbors

## Tech Stack

- Python
- pandas
- scikit-learn
- Jupyter Notebook

## Project Structure

```text
spam_classifier/
├── Untitled-1.ipynb
├── spam.csv
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/adipstha11/spam_classifier.git
   cd spam_classifier
   ```
2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn jupyter
   ```
4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `Untitled-1.ipynb` and run all cells from top to bottom.

## Results

All models were trained on the same train/test split and the same TF-IDF features for a fair comparison.

### Model Comparison

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 | False Positives (ham->spam) | True Negatives | False Negatives | True Positives |
|------|----------|----------------|-------------|---------|-------------------------------|----------------|-----------------|----------------|
| Linear SVM | 0.987444 | 0.977099 | 0.920863 | 0.948148 | 3 | 973 | 11 | 128 |
| Random Forest | 0.983857 | 0.991870 | 0.877698 | 0.931298 | 1 | 975 | 17 | 122 |
| Logistic Regression | 0.975785 | 0.982759 | 0.820144 | 0.894118 | 2 | 974 | 25 | 114 |
| K-Nearest Neighbors | 0.966816 | 1.000000 | 0.733813 | 0.846473 | 0 | 976 | 37 | 102 |
| Multinomial Naive Bayes | 0.962332 | 1.000000 | 0.697842 | 0.822034 | 0 | 976 | 42 | 97 |

### Best Models

- Best overall model: **Linear SVM**
- Best at minimizing false positives: **K-Nearest Neighbors** and **Multinomial Naive Bayes** (tie, both with 0 false positives)

## Why Some Models Performed Better Than Others

The performance differences are mainly due to how the SMS messages were represented. Since the dataset was transformed with TF-IDF, the feature space became high-dimensional and sparse. In this setting, linear models usually perform best.

That is why **Linear SVM** delivered the strongest overall results: it achieved the highest accuracy, the best spam recall, and the highest spam F1 score. This indicates it was the most effective model at separating spam from ham while maintaining a good balance between catching spam and avoiding errors.

**Logistic Regression** also performed well for similar reasons, but it missed more spam messages than Linear SVM.

**Random Forest** had slightly lower recall, but it produced very few false positives. This makes it a strong option when it is especially important to avoid flagging legitimate messages as spam.

**K-Nearest Neighbors** and **Multinomial Naive Bayes** achieved perfect spam precision and zero false positives, but they missed many more spam messages. As a result, their recall and overall balance were lower, suggesting they were too conservative for this task.

Overall, **Linear SVM** appears to be the most balanced model for this dataset, while **Random Forest** is a practical alternative when minimizing false positives is the top priority.

## Future Improvements

- Rename the notebook to something more descriptive (for example, `sms_spam_classifier.ipynb`)
- Add a `requirements.txt` file
- Save the best model and vectorizer with `joblib`
- Add a prediction script or simple API endpoint
- Add plots for side-by-side metric comparison

## Author

Adip Shrestha
