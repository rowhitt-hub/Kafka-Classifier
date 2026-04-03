

import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np


def load_file(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def build_models(kafka_path: str, non_kafka_path: str):
    kafka     = load_file("kafka.txt")
    non_kafka = load_file("non_kafka.txt")

    texts  = kafka + non_kafka
    labels = ["kafka"] * len(kafka) + ["non_kafka"] * len(non_kafka)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    nb = MultinomialNB(alpha=0.5)
    lr.fit(X, y)
    nb.fit(X, y)

    lr_cv = cross_val_score(lr, X, y, cv=min(5, len(kafka), len(non_kafka))).mean()
    nb_cv = cross_val_score(nb, X, y, cv=min(5, len(kafka), len(non_kafka))).mean()

    print(f"\n  Trained on {len(kafka)} Kafka / {len(non_kafka)} non-Kafka sentences")
    print(f"  CV Accuracy — Logistic Regression: {lr_cv:.1%}  |  Naive Bayes: {nb_cv:.1%}\n")

    return vectorizer, le, lr, nb


def classify(sentence: str, vectorizer, le, lr, nb) -> None:
    vec       = vectorizer.transform([sentence])
    kafka_idx = list(le.classes_).index("kafka")

    lr_proba  = lr.predict_proba(vec)[0][kafka_idx]
    nb_proba  = nb.predict_proba(vec)[0][kafka_idx]
    avg_proba = (lr_proba + nb_proba) / 2

    def bar(score):
        n = int(score * 28)
        return "█" * n + "░" * (28 - n)

    def verdict(score):
        return "🪲 KAFKA" if score >= 0.5 else "🌞 NOT KAFKA"

    print(f"\n  {'Model':<22} {'Verdict':<14} {'Kafka%':<8} Bar")
    print(f"  {'─'*22} {'─'*14} {'─'*8} {'─'*28}")
    print(f"  {'Logistic Regression':<22} {verdict(lr_proba):<14} {lr_proba:.1%}   [{bar(lr_proba)}]")
    print(f"  {'Naive Bayes':<22} {verdict(nb_proba):<14} {nb_proba:.1%}   [{bar(nb_proba)}]")
    print(f"  {'─'*22} {'─'*14} {'─'*8} {'─'*28}")
    print(f"  {'Ensemble avg':<22} {verdict(avg_proba):<14} {avg_proba:.1%}   [{bar(avg_proba)}]")
    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python kafka_classifier.py <kafka.txt> <non_kafka.txt>")
        sys.exit(1)

    vectorizer, le, lr, nb = build_models(sys.argv[1], sys.argv[2])

    print("=" * 60)
    print("  Kafka Classifier — type 'quit' to exit")
    print("=" * 60)

    while True:
        sentence = input("Enter a sentence: ").strip()
        if sentence.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not sentence:
            continue
        classify(sentence, vectorizer, le, lr, nb)


if __name__ == "__main__":
    main()

