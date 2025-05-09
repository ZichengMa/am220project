# AM220 Project: Representation Learning and Classification

## Project Structure
- `data_loader.py`: Contains functions to load and preprocess datasets (CIFAR-10 and MNIST).
- `train_eval.py`: Implements training and evaluation logic for classifiers.
- `representation_models.py`: Defines representation learning models (PCA, UMAP, Autoencoder, VAE) and classifiers.
- `evaluate_experiments_latent_d.py`: Evaluates how result(acc, training time) differs with varying latent dimensions.
- `evaluate_experiments.py`: Evaluates different representation learning methods.
## Setup
1. Install dependencies:
   Ensure `torch`, `torchvision`, `scikit-learn`, `umap-learn`, and `tqdm` are installed.

2. Download datasets (handled automatically by `data_loader.py`).

## Usage
### Running Experiments
1. To evaluate specific methods (e.g., UMAP) on a fixed latent dimension:
   ```bash
   python evaluate_experiments.py
   ```
2. To evaluate representation learning methods with varying latent dimensions:
   ```bash
   python evaluate_experiments_latent_d.py
   ```
   
### Modifying Parameters
- Adjust dataset (`DATASET`), latent dimensions (`latent_dims`), and epochs in the respective scripts.
- Add or remove methods in the `methods` dictionary in the evaluation scripts.

## Results
The results include metrics such as accuracy, F1 score, and computation times for representation learning and classifier training. These are printed in tabular format at the end of each evaluation script.

## Notes
- Ensure GPU support is enabled for faster training and evaluation.
- Use the `set_seed` function to ensure reproducibility.

