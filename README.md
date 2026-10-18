# Embedded LightGBM Feature Selection vs Mutual Information for NSL-KDD Intrusion Detection

Motivation: the paper "A CircleSA-Optimised feature selection and FNP-SE deep learning framework for enhanced network intrusion detection in enterprise systems" proposes a metaheuristic feature selection followed by a deep network for intrusion detection in enterprise systems. Its feature selection stage is the main lever for accuracy, so this experiment asks whether a simpler, embedded selector can match or beat a standard filter-based one.

Question: Can an embedded LightGBM-importance feature selection with a class-balanced LightGBM classifier beat a standard mutual-information selection with an MLP for binary intrusion detection on NSL-KDD?

Dataset: NSL-KDD, downloaded as raw text from the public defcom17/NSL_KDD GitHub mirror (KDDTrain+ for training, KDDTest+ as the held-out test split). This is not the paper's own dataset.

Baseline vs improved model:
- Baseline: top-k features by mutual information, then a scikit-learn MLPClassifier with two hidden layers.
- Improved: top-k features by LightGBM gain importance, then a class-balanced LightGBM classifier with fixed hyper-parameters.

Both models share the same loading, preprocessing, split, and evaluation code. The main metric is macro-F1 on the test split; accuracy and attack recall are secondary.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and figures/macro_f1.png).

## Reference

This project is motivated by the following paper:

> BB Gupta, SH Pan, A Gaurav, V Arya (2026). *A CircleSA-Optimised feature selection and FNP-SE deep learning framework for enhanced network intrusion detection in enterprise systems*. Enterprise Information. https://www.tandfonline.com/doi/abs/10.1080/17517575.2025.2595703

Paper author: Prof. Brij Bhooshan Gupta, Asia University (bbgupta@asia.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
