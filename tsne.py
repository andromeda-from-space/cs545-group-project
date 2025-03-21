
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing

DEF_PERPLEXITY = 30
DEF_LEARNING_RATE = 200
DEF_MAX_ITER = 1000
DIMENSIONS = 2

def run_tsne():
    # Modifiable hyperparameters for t-SNE
    perplexity = DEF_PERPLEXITY
    learning_rate = DEF_LEARNING_RATE
    max_iter = DEF_MAX_ITER

    # Get preprocessed data
    train_data, train_label, _, _ = preprocessing.main()
    
    target_column = np.argmax(train_label, axis=1)

    # Map numeric labels to class names
    class_names = ['dropout', 'enrolled', 'graduate']
    target_column_names = [class_names[label] for label in target_column]

    # Create dataframe with tsne results
    tsne_df = tsne(train_data, target_column_names, perplexity, learning_rate, max_iter)

    # Generate graph of tsne output
    display_results(tsne_df, perplexity, learning_rate, max_iter)


def tsne(X_scaled, target_column_names, perplexity, learning_rate, max_iter):
    # Perform t-SNE
    tsne = TSNE(n_components=DIMENSIONS, random_state=42, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create dataframe with t-SNE results and target variable
    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['Outcome'] = target_column_names  # Use class names instead of numeric labels
    return tsne_df


def display_results(tsne_df, perplexity, learning_rate, max_iter):

    title = f't-SNE Visualization\nPerplexity={perplexity}, Learning Rate={learning_rate}, Max Iter={max_iter}'
    filename = f't-SNE Perplexity={perplexity} Learning Rate={learning_rate} Max Iter={max_iter}'

    custom_palette = {
        'dropout': 'red',
        'graduate': 'green',
        'enrolled': 'blue'
    }

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue=tsne_df['Outcome'],  # Color-code by class names
        palette=custom_palette,
        data=tsne_df,
        legend='full',
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.legend(title='Outcome')
    plt.savefig(filename)
    plt.close()
    #plt.show()


if __name__ == "__main__":
    run_tsne()