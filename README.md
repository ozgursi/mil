# Prototype-based Multiple Instance Learning (MIL)

A prototype-based decision tree and random forest framework for Multiple Instance Learning (MIL) classification. In MIL, data is organized into "bags" of instances where only the bag-level label is known — individual instance labels are not available.

## How It Works

Each tree node learns (or randomly selects) a **prototype** — a representative pattern. Instances in each bag are compared to the prototype using distance metrics (min, max, mean), and these distance-based features are aggregated at the bag level. A decision tree stump then finds the best split on these features. The process repeats recursively to build a full tree. Multiple trees form a random forest with majority voting.

When `use_prototype_learner=True`, prototypes are learned via gradient descent using a neural network (`ShapeletGenerator`) with BCE loss and regularization. When `False`, prototypes are randomly sampled from the training data.

## Main Files

| File | Description |
|------|-------------|
| [prototype.py](prototype.py) | **Main script.** Contains the full pipeline: data loading, prototype learning (`find_prototype`), `PrototypeTreeClassifier`, `PrototypeForest`, and the experiment runner with cross-validation. |
| [model.py](model.py) | Original `ShapeletGenerator` PyTorch module. Redefined inside `prototype.py` with an updated API. |
| [best_params.csv](best_params.csv) | Hyperparameter configuration (max_depth, ntree, PCA settings) per dataset. |
| [utils.py](utils.py) | Shared utilities: `convert_to_bags`, `load_data`, `plot_prototypes`. |

## Algorithmic Variants

| File | Difference from prototype.py |
|------|------------------------------|
| [linear-tree.ipynb](linear-tree.ipynb) | Uses `LogisticRegression` for node splits instead of decision tree stumps. No prototype learning. |
| [linear-learning.ipynb](linear-learning.ipynb) | Uses a learned neural net linear model (`CustomModel`) for node splits. |
| [prototype_regression.ipynb](prototype_regression.ipynb) | Regression adaptation: `DecisionTreeRegressor`, MAE metric, mean predictions instead of class probabilities. |
| [prototype_regression-learning.ipynb](prototype_regression-learning.ipynb) | Regression with prototype learning: `MSELoss`, early stopping on loss. |

## Superseded / Development Files

These are older versions or development snapshots that have been incorporated into `prototype.py`:

| File | Notes |
|------|-------|
| [prototype_forest.py](prototype_forest.py) | Old standalone tree/forest classes. Instance-level splitting (superseded by bag-level in prototype.py). |
| [run2.py](run2.py) | GPyOpt Bayesian optimization script. Superseded by `best_params.csv`. |
| [forest_updated.ipynb](forest_updated.ipynb) | Oldest notebook version with GPyOpt integration. |
| [forest-current.ipynb](forest-current.ipynb) | Development snapshot with debug breakpoints. |

## Data Layout

- `datasets/` — Classification datasets (CSV format: label, bag_id, features...)
- `datasets_regression/` — Regression datasets (space-separated)
- `performance_x/` — Output results for classification experiments
- `performance_linear/` — Output results for linear-tree variant

## Dependencies

- Python 3
- PyTorch
- NumPy, Pandas, SciPy
- scikit-learn (DecisionTree, PCA, StandardScaler, AUC)
- matplotlib

## Known Issues in prototype.py

1. **Missing `zero_grad()`** in `find_prototype` — gradients accumulate across training steps
2. **Shadowed loop variable** — eval loop reuses variable `i`, corrupting the outer training counter
3. **Prototype reference not cloned** — `best_prototype = model.prototypes` stores a reference, not a copy

These should be addressed before relying on `use_prototype_learner=True` results.
