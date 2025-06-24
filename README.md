# Spam Classifier

A machine learning project for classifying text messages as spam or not spam, using a combination of feature extraction techniques, decision trees, and custom data processing pipelines.

---

## Features

- Preprocessing of raw text data (`spam.csv`)
- Feature extraction using:
  - Bag-of-Words
  - TF-IDF
- Decision Tree Classifier (custom implementation)
- Model evaluation and logging
- Modular code design
- Jupyter notebook for exploratory data analysis (`EDA.ipynb`)

---

## Project Structure

```

spam-classifier/
├── data/                       # Processed and raw datasets
├── utils/                      # Utility functions (logging, plotting)
├── **pycache**/               # Cached Python bytecode
├── config.py                   # Configuration settings and paths
├── data\_io.py                  # Data loading/saving
├── feature\_extraction.py       # Text vectorization (TF-IDF, etc.)
├── processing.py               # Preprocessing functions
├── decision\_tree.py            # Custom decision tree implementation
├── model\_runner.py             # Training + evaluation logic
├── main.py                     # Entry point for training/evaluation
├── EDA.ipynb                   # Notebook for data exploration
├── requirements.txt            # (Optional) Python dependencies
└── README.md

````

---

## Usage

### Run the project:
```bash
python main.py
````

This will:

* Load and preprocess the dataset
* Extract features
* Train a decision tree
* Output classification results and logs

---

## EDA Notebook

The [`EDA.ipynb`](EDA.ipynb) notebook provides data exploration using `pandas`, visualizations with `matplotlib`/`seaborn`, and summary statistics of the features.

---

## Dependencies

Make sure to install the necessary packages. If using `conda`, create an environment and install:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

* Add cross-validation and parameter tuning
* Integrate more classifiers (e.g., SVM, Naive Bayes)
* Export model artifacts
* Create a simple web interface for predictions

---

## Author

Created by [shlomias1](https://github.com/shlomias1)

---

## License

This project is open-source and free to use under the [MIT License](LICENSE).

```

---
