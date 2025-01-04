# AI-Algorithms-Made-Easy

**Under Development**

![Under Development](under_development.png?raw=true "Under Development")

Welcome to **AI-Algorithms-Made-Easy**! This project is a comprehensive collection of artificial intelligence algorithms implemented from scratch using **PyTorch**. Our goal is to demystify AI by providing clear, easy-to-understand code and detailed explanations for each algorithm.

Whether you're a beginner in machine learning or an experienced practitioner, this project offers resources to enhance your understanding and skills in AI.

---

## Project Description

**AI-Algorithms-Made-Easy** aims to make AI accessible to everyone by:

- **Intuitive Implementations**: Breaking down complex algorithms into understandable components with step-by-step code.
- **Educational Notebooks**: Providing Jupyter notebooks that combine theory with practical examples.
- **Interactive Demos**: Offering user-friendly interfaces built with **Gradio** to experiment with algorithms in real-time.
- **Comprehensive Documentation**: Supplying in-depth guides and resources to support your AI learning journey.

Our mission is to simplify the learning process and provide hands-on tools to explore and understand AI concepts effectively.

---

## Table of Contents

- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Algorithms Implemented

*This project is currently under development. Stay tuned for updates!*

### Supervised Learning (Scikit-Learn)
#### 1. Regression ([Documentation](docs/Regression_Documentation.md), [Interface](https://huggingface.co/spaces/mboukabous/train_regression), [Notebook](notebooks/Train_Supervised_Regression_Models.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mboukabous/AI-Algorithms-Made-Easy/blob/main/notebooks/Train_Supervised_Regression_Models.ipynb))
- [Linear Regression](models/supervised/regression/linear_regression.py)
- [Ridge Regression](models/supervised/regression/ridge_regression.py)
- [Lasso Regression](models/supervised/regression/lasso_regression.py)
- [ElasticNet Regression](models/supervised/regression/elasticnet_regression.py)
- [Decision Tree](models/supervised/regression/decision_tree_regressor.py)
- [Random Forest (Bagging)](models/supervised/regression/random_forest_regressor.py)
- [Gradient Boosting (Boosting)](models/supervised/regression/gradient_boosting_regressor.py)
- [AdaBoost (Boosting)](models/supervised/regression/adaboost_regressor.py)
- [XGBoost (Boosting)](models/supervised/regression/xgboost_regressor.py)
- [LightGBM](models/supervised/regression/lightgbm_regressor.py)
- [CatBoost](models/supervised/regression/catboost_regressor.py)
- [Support Vector Regressor (SVR)](models/supervised/regression/support_vector_regressor.py)
- [K-Nearest Neighbors (KNN) Regressor](models/supervised/regression/knn_regressor.py)
- [Extra Trees Regressor](models/supervised/regression/extra_trees_regressor.py)
- [Multilayer Perceptron (MLP) Regressor](models/supervised/regression/mlp_regressor.py)

#### 2. Classification ([Documentation](docs/Classification_Documentation.md), [Interface](https://huggingface.co/spaces/mboukabous/train_classificator), [Notebook](notebooks/Train_Supervised_Classification_Models.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mboukabous/AI-Algorithms-Made-Easy/blob/main/notebooks/Train_Supervised_Classification_Models.ipynb))
- [Logistic Regression](models/supervised/classification/logistic_regression.py)
- [Decision Tree Classifier](models/supervised/classification/decision_tree_classifier.py)
- [Random Forest Classifier (Bagging)](models/supervised/classification/random_forest_classifier.py)
- [Extra Trees Classifier](models/supervised/classification/extra_trees_classifier.py)
- [Gradient Boosting Classifier (Boosting)](models/supervised/classification/gradient_boosting_classifier.py)
- [AdaBoost Classifier (Boosting)](models/supervised/classification/adaboost_classifier.py)
- [XGBoost Classifier (Boosting)](models/supervised/classification/xgboost_classifier.py)
- [LightGBM Classifier (Boosting)](models/supervised/classification/lightgbm_classifier.py)
- [CatBoost Classifier (Boosting)](models/supervised/classification/catboost_classifier.py)
- [Support Vector Classifier (SVC)](models/supervised/classification/svc.py)
- [K-Nearest Neighbors (KNN) Classifier](models/supervised/classification/knn_classifier.py)
- [Multilayer Perceptron (MLP) Classifier](models/supervised/classification/mlp_classifier.py)
- [GaussianNB (Naive Bayes Classifier)](models/supervised/classification/gaussian_nb.py)
- [Linear Discriminant Analysis (LDA)](models/supervised/classification/linear_discriminant_analysis.py)
- [Quadratic Discriminant Analysis (QDA)](models/supervised/classification/quadratic_discriminant_analysis.py)

 
### Unsupervised Learning (Scikit-Learn) ([Documentation](docs/Unsupervised_Documentation.md), [Interface](https://huggingface.co/spaces/mboukabous/train_unsupervised), [Notebook](notebooks/Train_Unsupervised_Models.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mboukabous/AI-Algorithms-Made-Easy/blob/main/notebooks/Train_Unsupervised_Models.ipynb))
#### 1. Clustering
- [K-Means](models/unsupervised/clustering/kmeans.py)
- [DBSCAN](models/unsupervised/clustering/dbscan.py)
- [Gaussian Mixture](models/unsupervised/clustering/gaussian_mixture.py)
- [Hierarchical Clustering](models/unsupervised/clustering/hierarchical_clustering.py)

#### 2. Dimensionality Reduction
- [Principal Component Analysis (PCA)](models/unsupervised/dimred/pca.py)
- [t-SNE](models/unsupervised/dimred/tsne.py)
- [UMAP](models/unsupervised/dimred/umap.py)
  
#### 3. Anomaly (Outlier) Detection
- [Isolation Forest](models/unsupervised/anomaly/isolation_forest.py)
- [One-Class SVM](models/unsupervised/anomaly/one_class_svm.py)
- [Local Outlier Factor (LOF)](models/unsupervised/anomaly/local_outlier_factor.py)

### Computer Vision
#### 1. Image Classification
- Convolutional Neural Networks (CNN)
- Example CNN Architecture: TinyVGG (from [CNN Explainer](https://poloclub.github.io/cnn-explainer/))
- Transfer Learning (using [TorchVision](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights))
  
#### 2. Object Detection
- Faster R-CNN
- YOLO (You Only Look Once)
- SSD (Single Shot MultiBox Detector)

#### 3. Semantic Segmentation
- U-Net
- DeepLab
- PSPNet

#### 4. Style Transfer

#### 5. Image Captioning
- CNN + RNN approach (or CNN + Transformer)
- Potential integration with NLP techniques

#### 6. Generative Models (Vision)
- DCGAN (Deep Convolutional Generative Adversarial Networks)
- StyleGAN
- Diffusion Models

#### 7. Self-Supervised Learning
- SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
- BYOL (Bootstrap Your Own Latent)
- SwAV (Swapping Assignments Between Views)
- DINO (Self-Distillation with No Labels)
- CLIP (Contrastive Language–Image Pre-training)

### Natural Language Processing (NLP)
#### 1. Sequence Models
- RNN (Vanilla Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

#### 2. Transformers
- Encoder-Decoder models (e.g., BERT, GPT, etc.)
- Attention Mechanisms

#### 3. Core NLP Tasks
- Text Classification (sentiment analysis, topic classification)
- Machine Translation
- Named Entity Recognition (NER)
- Text Summarization (extractive or abstractive)
- Question Answering
- Language Modeling (causal or masked)

#### 4. Generative Models (Text)
- Seq2Seq with attention
- GPT-like for text generation

### Time Series Analysis
- Time Series Forecasting with RNNs
- Temporal Convolutional Networks (TCN)
- Transformers for Time Series

### Reinforcement Learning
#### 1. Value-Based Methods
Q-Learning
Deep Q-Networks (DQN)
#### 2. Policy-Based Methods
REINFORCE (Policy Gradients)
Actor-Critic (A2C, PPO, etc.)
#### 3. Advanced RL Topics
Hierarchical RL
Multi-Agent RL
Offline RL (Batch RL)

---

## Project Structure

- **models/**: Contains all the AI algorithm implementations, organized by category.
- **data/**: Includes datasets and data preprocessing utilities.
- **utils/**: Utility scripts and helper functions.
- **scripts/**: Executable scripts for training, testing, and other tasks.
- **interfaces/**: Interactive applications using Gradio and web interfaces.
- **notebooks/**: Jupyter notebooks for tutorials and demonstrations.
- **deploy/**: Scripts and instructions for deploying models.
- **website/**: Files related to the project website.
- **docs/**: Project documentation.
- **examples/**: Example scripts demonstrating how to use the models.

---

## Installation

*Installation instructions will be provided once the initial release is available.*

---

## Usage

*Usage examples and tutorials will be added as the project develops.*

---

## Contributing

We welcome contributions from the community! To contribute:

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine.
3. **Create a new branch** for your feature or bug fix.
4. **Make your changes** and commit them with descriptive messages.
5. **Push your changes** to your forked repository.
6. **Open a pull request** to the main repository.

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or feedback:

- **GitHub Issues**: Please open an issue on the [GitHub repository](https://github.com/mboukabous/AI-Algorithms-Made-Easy/issues).
- **Email**: You can reach us at [m.boukabous95@gmail.com](mailto:m.boukabous95@gmail.com).

---

*Thank you for your interest in **AI-Algorithms-Made-Easy**! We are excited to build this resource and appreciate your support and contributions.*

---

## Acknowledgments

- **PyTorch**: For providing an excellent deep learning framework.
- **Gradio**: For simplifying the creation of interactive demos.
- **OpenAI's ChatGPT**: For assistance in planning and drafting project materials.

---

## Stay Updated

- **Watch** this repository for updates.
- **Star** the project if you find it helpful.
- **Share** with others who might be interested in learning AI algorithms.

---

*Let's make AI accessible and easy to learn for everyone!*
