
# League of Legends Match Outcome Prediction

A machine learning project that uses **PyTorch logistic regression** to predict League of Legends match outcomes from in-game statistics. The project covers the complete machine learning workflow, including data preprocessing, model training, regularization, evaluation, hyperparameter tuning, and feature importance analysis.

## Dataset

The dataset (`league_of_legends_data_large.csv`) contains eight numerical features describing match performance:

- **Combat:** Kills, deaths, assists, and damage dealt
- **Economy:** Gold earned and creep score (CS)
- **Vision:** Wards placed and wards killed

The target variable, `win`, represents the binary match outcome (0 = Loss, 1 = Win).

## Methodology

The project is implemented in eight stages:

1. **Data Preprocessing:** Split the dataset into 80% training and 20% testing using `train_test_split`. Standardize features with `StandardScaler` and convert the data into PyTorch tensors.

2. **Model Architecture:** Implement logistic regression using `nn.Linear` with a sigmoid activation function to produce binary classification probabilities.

3. **Model Training:** Train the model for 1,000 epochs using Binary Cross-Entropy Loss (`BCELoss`), stochastic gradient descent (`SGD` optimizer), and backpropagation.

4. **Regularization:** Apply L2 regularization (`weight_decay=0.01`) to penalize large weights and help reduce overfitting.

5. **Model Evaluation:** Evaluate classification accuracy, precision, recall, and F1-score. Generate confusion matrices and ROC curves to examine predictive performance.

6. **Model Persistence:** Save the trained model parameters using `torch.save()` and reload them using `torch.load()`. Verify that the reloaded model retains its accuracy.

7. **Hyperparameter Tuning:** Compare learning rates of 0.01, 0.05, and 0.1 using 100 training epochs per configuration.

8. **Feature Importance:** Extract the learned logistic regression coefficients, rank features by absolute weight, and visualize their contributions using a horizontal bar chart.

## Results

| Evaluation | Test Accuracy |
|---|---:|
| Initial Logistic Regression | 52.00% |
| L2-Regularized Model (Initial Run) | 51.50% |
| Retrained and Saved Model | 49.50% |
| Best Learning Rate Configuration (100 Epochs) | 50.00% |

- **Best tested learning rate:** 0.01
- **ROC-AUC:** Approximately 0.50
- **Highest-weight features:** Gold earned, kills, and deaths

The model achieved performance close to random classification, highlighting the limitations of linear decision boundaries for the available match statistics. Feature importance analysis provides insight into how individual statistics influence model predictions.

## Technologies

- **Python**
- **PyTorch:** Model implementation, automatic differentiation, and optimization
- **Pandas / NumPy:** Data manipulation and numerical operations
- **Scikit-learn:** Preprocessing, dataset splitting, and evaluation metrics
- **Matplotlib / Seaborn:** Performance visualization
- **Jupyter Notebook:** Interactive development and experimentation

## Repository Structure

- `main.ipynb` — Complete implementation, experiments, evaluation, and visualizations
- `league_of_legends_data_large.csv` — Match statistics dataset
- `lol_logistic_regression.pth` — Saved weights of the trained L2-regularized model
- `README.md` — Project documentation

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Helia-Karisani/league_of_legends_match_estimator.git
cd league_of_legends_match_estimator
```

Install the required dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

Open `main.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially.

The notebook generates all evaluation plots directly, so no separate image files are required.

## Limitations

The current model demonstrates limited predictive performance. Hyperparameter selection was performed using test accuracy rather than a separate validation set. Additionally, the saved `.pth` file contains model parameters only; the fitted feature scaler must also be preserved for consistent predictions on new data.
