# Bank Note Authentication using Random Forest Classifiers

## Introduction
Counterfeit currency detection is a critical task for national financial security. The CIA Banknote Authentication dataset provides real-world data extracted from genuine and forged banknotes, captured using image processing techniques. This dataset allows for the application of machine learning algorithms to classify banknotes as either authentic or fake based on statistical features derived from scanned images.

The dataset was originally provided by the Central Intelligence Agency (CIA) and is widely used for benchmarking binary classification models.

### Dataset Summary
The dataset consists of 1,372 observations and 5 columns (4 features + 1 label), where each row represents statistical data extracted from the image of a banknote.

## Features
| Column Name | Description                                                                  |
| ----------- | ---------------------------------------------------------------------------- |
| `variance`  | Variance of the wavelet-transformed image (a measure of intensity spread).   |
| `skewness`  | Skewness of the wavelet-transformed image (asymmetry of pixel distribution). |
| `curtosis`  | Kurtosis of the wavelet-transformed image (sharpness of the peak).           |
| `entropy`   | Entropy of the image (amount of randomness or texture irregularity).         |
| `class`     | Label: `0` = forged banknote, `1` = genuine banknote.                        |

Each feature was extracted using wavelet transform—a mathematical tool used for image compression and feature extraction. The simplicity and effectiveness of the dataset make it ideal for classification model benchmarking, including logistic regression, SVM, random forest, and neural networks.

## Methodology 
1. Initial data description was obtained using ```df.head```, ```df.describe``` and ```df.info```. These provided a brief view of the entire dataset, along with statistical analyses of the different data features and the different data types.
2. A pairplot was used to get the different pairwise analyses as one single figure.
3. The data was split into a 85:15 ratio, for training and testing respectively.
4. ```GridSearch``` was utilized to test for different values of different hyperparameters, and a Random Forest Classifier was utilized for model training.
5. Model was evaluated using a Classification report, and confusion matrix.
6. The model was explained using ```LIME```.

## Results 
* The confusion matrix reported a fantastic 205 true values with one single false value.

This classification report shows that the model performs exceptionally well in distinguishing between forged (class 0) and genuine (class 1) banknotes. It achieves 100% accuracy, correctly classifying all 206 test samples except for possibly one. Precision and recall are both very high for both classes—1.00 for class 0 and around 0.99–1.00 for class 1—indicating minimal false positives and false negatives. The F1-scores, which balance precision and recall, are also close to perfect. Overall, the model demonstrates excellent and reliable performance on the test data.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.99   | 1.00     | 124     |
| 1     | 0.99      | 1.00   | 0.99     | 82      |

**Accuracy:** 1.00 (on 206 samples)

| Metric      | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Macro Avg   | 0.99      | 1.00   | 0.99     | 206     |
| Weighted Avg| 1.00      | 1.00   | 1.00     | 206     |

### Explanation using LIME 
* The model predicted the banknote as forged (class 0) with a probability of 56.6% (1 - 0.4338), which matches the true label (Right: 0.0).
* Variance ≤ -1.76 and Entropy ≤ -2.38 positively influenced the prediction toward genuine (class 1), shown by the green bars.
* Skewness > 6.73 had the strongest negative impact, pushing the model toward predicting forged (class 0), followed by Curtosis between 0.65 and 3.36.
* The red bars represent features that reduced the likelihood of classifying the note as genuine.
* Overall, despite some features favoring genuineness, the model correctly predicted the note as forged due to dominant negative feature effects.

The LIME plot can be found in the Jupyter Notebook 
