from data_loader import load_and_prepare_data
from visualization import run_all_visualizations
from model_training import train_all_models

# 1. Încarcă datele
train_data, X, y = load_and_prepare_data('winequality-red.csv')

# 2. Rulează vizualizările
run_all_visualizations(train_data)

# 3. Rulează antrenarea modelelor
train_all_models(X, y)
