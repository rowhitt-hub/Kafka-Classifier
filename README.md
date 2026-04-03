# Kafka Classifier

This project provides a Python-based utility to classify whether a given sentence stylistically resembles the writing of Franz Kafka. It utilizes machine learning models trained on specific text datasets to provide a probabilistic "Kafkaesque" score.

---

## Features

* [cite_start]**Dual-Model Analysis**: Uses both Logistic Regression and Multinomial Naive Bayes to evaluate text[cite: 27, 28].
* [cite_start]**Ensemble Scoring**: Calculates an average probability (Ensemble avg) across both models for a more balanced verdict[cite: 47, 59].
* [cite_start]**Real-time Classification**: Provides an interactive command-line interface for testing individual sentences[cite: 68, 70].
* [cite_start]**Visual Feedback**: Displays results in a formatted table including a percentage score and a visual progress bar representing the Kafka confidence level[cite: 53, 55].

---

## Technical Implementation

### Machine Learning Pipeline
1.  [cite_start]**Text Vectorization**: Converts raw text into numerical features using `TfidfVectorizer` with support for both unigrams and bigrams ($ngram\_range=(1,2)$)[cite: 23, 24].
2.  [cite_start]**Label Encoding**: Transforms categorical labels ("kafka" vs "non_kafka") into a numerical format suitable for model training[cite: 25, 26].
3.  [cite_start]**Cross-Validation**: Evaluates model performance during training using `cross_val_score` to ensure reliability[cite: 31, 34].

### Dependencies
* [cite_start]`scikit-learn`: Used for vectorization, label encoding, and classification models[cite: 5, 6, 7, 8].
* [cite_start]`numpy`: Used for numerical operations[cite: 9].
* [cite_start]`pathlib`: Used for file path handling and reading input data[cite: 4, 12].

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
[cite_start][cite: 63]

### Interactive Mode
[cite_start]Once launched, the program will display the training accuracy for both models (e.g., Logistic Regression: 92.5%, Naive Bayes: 90.1%) and prompt for input[cite: 83, 84].

* [cite_start]**Enter a sentence**: Type any sentence to receive a verdict[cite: 70].
* [cite_start]**Exit**: Type `quit`, `exit`, or `q` to close the program[cite: 71].

---

## Example Output

When a sentence is entered, the tool outputs a table similar to the following:

| Model | Verdict | Kafka% | Bar |
| :--- | :--- | :--- | :--- |
| Logistic Regression | KAFKA | 64.3% | [##########] |
| Naive Bayes | KAFKA | 76.3% | [############] |
| Ensemble avg | KAFKA | 70.3% | [###########] |

[cite_start][cite: 131, 132, 133, 134, 135, 136, 138, 139, 140]
