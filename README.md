# Kafka Classifier

This project provides a Python-based utility to classify whether a given sentence stylistically resembles the writing of Franz Kafka. It utilizes machine learning models trained on specific text datasets to provide a probabilistic "Kafkaesque" score.

---

## Features

* **Dual-Model Analysis**: Uses both Logistic Regression and Multinomial Naive Bayes to evaluate text.
* **Ensemble Scoring**: Calculates an average probability (Ensemble avg) across both models for a more balanced verdict.
* **Real-time Classification**: Provides an interactive command-line interface for testing individual sentences.
* **Visual Feedback**: Displays results in a formatted table including a percentage score and a visual progress bar representing the Kafka confidence level.

---

## Technical Implementation

### Machine Learning Pipeline
1.  **Text Vectorization**: Converts raw text into numerical features using `TfidfVectorizer` with support for both unigrams and bigrams ($ngram\_range=(1,2)$).
2.  **Label Encoding**: Transforms categorical labels ("kafka" vs "non_kafka") into a numerical format suitable for model training.
3.  **Cross-Validation**: Evaluates model performance during training using `cross_val_score` to ensure reliability.

### Dependencies
* `scikit-learn`: Used for vectorization, label encoding, and classification models.
* `numpy`: Used for numerical operations.
* `pathlib`: Used for file path handling and reading input data.

---

## Usage

### Training the Models
The script requires two text files to build the classification models:
1.  A file containing sentences written by Kafka.
2.  A file containing non-Kafka sentences.

To run the classifier from the command line:
```bash
python kafka_classifier.py <kafka.txt> <non_kafka.txt>
```


### Interactive Mode
Once launched, the program will display the training accuracy for both models (e.g., Logistic Regression: 92.5%, Naive Bayes: 90.1%) and prompt for input

* **Enter a sentence**: Type any sentence to receive a verdict.
* **Exit**: Type `quit`, `exit`, or `q` to close the program.

---

## Example Output

When a sentence is entered, the tool outputs a table similar to the following:

| Model | Verdict | Kafka% | Bar |
| :--- | :--- | :--- | :--- |
| Logistic Regression | KAFKA | 64.3% | [##########] |
| Naive Bayes | KAFKA | 76.3% | [############] |
| Ensemble avg | KAFKA | 70.3% | [###########] |


