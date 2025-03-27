import matplotlib.pyplot as plt
import seaborn as sns

def run_all_visualizations(train_data):
    train_data.hist(bins=15, figsize=(30, 20), layout=(4, 3))
    plt.suptitle("Distributia valorilor pe caracteristici")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_data, x='quality', y='alcohol')
    plt.title("Distrib alcolului in functie de calitatea vinurlui")
    plt.show()

    selected = ['alcohol', 'volatile acidity', 'sulphates', 'pH', 'density', 'quality']
    sns.pairplot(train_data[selected], hue='quality', palette="husl", height=2.5, aspect=1.5)
    plt.tight_layout()
    plt.show()

    sns.countplot(x='quality', data=train_data, palette='viridis')
    plt.title("Distribuția scorurilor de calitate a vinului")
    plt.show()
