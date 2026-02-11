# IMDB Sentiment Analysis with PyTorch

An end-to-end deep learning pipeline built to classify sentiment in 50,000 movie reviews. This project demonstrates production-ready PyTorch workflows, including custom data orchestration, neural network regularization, and rigorous evaluation.

## Project Overview
The goal was to build a robust binary classifier to distinguish between positive and negative movie reviews. Unlike basic implementations, this project focuses on preventing data leakage and optimizing memory usage through custom PyTorch **Datasets** and **DataLoaders**.

* **Final Accuracy:** 87.6%
* **Dataset:** IMDB 50K Movie Reviews

## Tech Stack
* **Deep Learning:** PyTorch (Neural Networks, DataLoaders, SGD Optimization)
* **NLP and Preprocessing:** Scikit-learn (CountVectorizer), Regex (re), HTML cleaning
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib (Loss and Accuracy curves)

## Architecture and Implementation
* **Custom Neural Network:** A 3-layer architecture (1000 to 64 to 1) utilizing **ReLU** activation functions.
* **Regularization:** Implemented **Dropout (p=0.2)** to mitigate overfitting, ensuring the model generalizes well to unseen reviews.
* **Vectorization:** Employed a **Bag-of-Words (BoW)** strategy restricted to the top 2,000 features to maintain a balance between computational efficiency and model signal.
* **Evaluation:** Monitored **Binary Cross-Entropy (BCE) Loss** across 20 training epochs to ensure convergence.

## Results
The model achieved a stable **87.6% accuracy** on the test set. By plotting training vs. validation loss, I confirmed that the Dropout layers effectively bridged the gap between training performance and real-world generalization.

## Reflection: The Engineering Perspective

### What Went Well
* **Data Orchestration:** Leveraging **PyTorch DataLoaders** allowed for memory-efficient batching (Batch Size: 64), which is critical for scaling to larger datasets in a production environment.
* **Feature Engineering:** By fitting the **CountVectorizer** only on the training split, I successfully avoided information leakage, a common pitfall in NLP pipelines.

### Challenges Faced and Solved
* **Overfitting:** Initial iterations showed high training accuracy but lower validation performance. I resolved this by introducing **Dropout layers** and fine-tuning the **Learning Rate (0.01)** with an SGD optimizer.
* **Text Noise:** The raw IMDB data contained HTML tags and special characters. I engineered a custom cleaning function using **regex** to normalize the input before vectorization.
