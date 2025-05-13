# Yandex-CV-Training


## NoPyTorch.py (https://contest.yandex.ru/contest/75234/problems/A/)

**NumPy-based deep learning framework from scratch, mimicking the PyTorch API.**

It includes:

*   A base `Module` class for all layers and a `Sequential` container for model construction.
*   Essential layers: `Linear`, `SoftMax`, `LogSoftMax`, `BatchNormalization`, `Dropout`.
*   Common activation functions: `ReLU`, `LeakyReLU`, `ELU`, `SoftPlus`.
*   Loss functions (Criterions): `MSECriterion`, `ClassNLLCriterion`.
*   Optimizers: `SGD with momentum`, `Adam`.
*   Basic CNN layers: `Conv2d`, `MaxPool2d`, `Flatten`.


## Pistachios (drones).ipynb (https://contest.yandex.ru/contest/75232/problems/A/)

**Approaches Tried:**

1.  **Exploratory Data Analysis (EDA)**
2.  **Baseline Model Evaluation:**
    *   Standard classifiers (LR, LDA, KNN, CART, NB, SVM) on raw data.
3.  **Model Evaluation with Preprocessing:**
    *   Standard classifiers with `StandardScaler` and `PCA(n_components=8)`.
4.  **Hyperparameter Tuning (GridSearchCV):**
    *   SVC on scaled data (Best performing: **0.873786 CV Accuracy**).
    *   KNN on raw data.
    *   Logistic Regression on raw data.
5.  **Ensemble Methods (Attempted, Incomplete):**
    *   Pipelines with PCA and AdaBoost, GBM, Random Forest, Extra Trees
6.  **Final Prediction:**
    *   Using the best-tuned SVC model on scaled test data.

## image_similarity.ipynb (https://contest.yandex.ru/contest/75233/problems/B/)

**Approaches:**

1. **Prepare image (Crop & Resize)**
2. **Extract features (Last-layer-freezed ResNet50)**
3. **Find similarity**

## hw_sobel_and_simple_hog.ipynb (https://contest.yandex.ru/contest/75229/problems/A/)

## hw_panorama_matching_.ipynb (https://contest.yandex.ru/contest/75230/problems/)


## parkinson_prediction.ipynb

**Task:** Using patient parameters (gender, age, and Parkinson's type) along with time series data of hand movements, predict which standard Parkinson's test the patient is executing.

**Approache:** Hybrid CNN-RNN network + Attention layer.
