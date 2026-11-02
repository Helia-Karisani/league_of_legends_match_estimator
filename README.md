# League of Legends Match Outcome Prediction

A machine learning project implementing **logistic regression in PyTorch** to predict League of Legends match outcomes using in-game statistics.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, regularization, evaluation, hyperparameter tuning, and feature importance analysis.

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

3. **Model Training:** Train the model for 1,000 epochs using Binary Cross-Entropy Loss (`BCELoss`), full-batch gradient updates with PyTorch's `SGD` optimizer, and backpropagation.

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

### Interpretation of Results

The model achieved approximately 50–52% test accuracy, indicating limited predictive performance. Several observations help explain these findings:

**Limited learning on training data.** The initial training accuracy was approximately 53.62%, compared with 52.00% on the test set. The relatively low performance on both sets suggests that the model struggled to learn useful predictive relationships, rather than primarily suffering from overfitting.

**Weak class discrimination.** The ROC-AUC of approximately 0.50 indicates that the model was unable to reliably distinguish wins from losses. Furthermore, the test set contained 105 wins and 95 losses, meaning a classifier that always predicted a win would achieve 52.5% accuracy. The trained models did not outperform this simple baseline.

**Potential limitations of the features and model.** Logistic regression assumes a linear relationship between the input features and the log-odds of winning. Match outcomes may depend on complex interactions among player performance, team composition, objectives, and game progression that are not captured by the available features or a linear model. Alternatively, the dataset may contain weak or noisy relationships between its features and labels. These are plausible explanations, not conclusions established by the experiments.

**Limited benefit from optimization.** L2 regularization and learning-rate tuning did not substantially improve predictive performance. Regularization primarily controls model complexity and cannot compensate for insufficient predictive information. Similarly, adjusting the learning rate affects optimization but does not increase the expressive capacity of logistic regression.

**Feature importance limitations.** Gold earned, kills, and deaths had the largest absolute learned coefficients. However, given the model's weak overall predictive performance, these coefficients should be interpreted cautiously rather than as definitive evidence of which statistics determine match outcomes.

### Conclusion

The experiments demonstrate a complete PyTorch binary classification workflow, although the resulting model does not provide reliable match predictions.

The project emphasizes model implementation, systematic evaluation, and interpretation of machine learning results. The findings illustrate an important practical lesson: completing model training and optimization does not necessarily guarantee predictive performance.

Further investigation could involve examining feature-label relationships, incorporating additional match statistics, comparing nonlinear classifiers, and evaluating models against baseline predictions using a separate validation set.

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

The current model demonstrates limited predictive performance, and the exact cause has not been established through additional data-quality or model-comparison experiments.

Hyperparameter selection was performed using test accuracy rather than a separate validation set. Additionally, the saved `.pth` file contains model parameters only; the fitted feature scaler must also be preserved for consistent predictions on new data.
