# PurLock

This repository provides the artifacts for our ACM CCS 2026 submission, in compliance with the Open Science Policy.  

## Contents

- **BERT Models (`bert/`)**: Scripts for text classification experiments.  
- **CNN Models (`cnn/`)**: Scripts for image classification experiments.  
- **LSTM Models (`lstm/`)**: Scripts for time-series regression experiments.  

Each folder includes:
- `*_baseline.py` – Standard model implementation without protection.  
- `*_DP.py` – Model with DP-SGD applied.   
- `*_purlock.py` – Model with our proposed PurLock method.

For specificity, the demo code for localization task is in **`unauthorized_task/`**, which includes:
- `*_baseline.py` – Standard model implementation without protection.
- `*_DP_purlock.py` – Model with our proposed PurLock method and DP-SGD.

## Datasets 

- **Image Classification (CNN-based models)**  
  - CIFAR-10: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)  
  - Street View House Numbers (SVHN): [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)  

- **Text Sentiment Analysis (BERT-based models)**  
  - Rotten Tomatoes Sentiment Dataset (SST): [https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)  
  - Internet Movie Database (IMDb): [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)  

- **Stock Price Prediction (RNN/LSTM models)**  
  - Apple Historical Stock Price (AAPL): Publicly available from financial data sources such as [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history/).  



Artifacts are shared **for availability only** at the review stage.   
