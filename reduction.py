import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

train_path = "./data/train_subset_10.csv"
val_path = "./data/val_subset_10.csv"
test_path = "./data/test_subset_10.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

def get_X_y_id(path: str, df: pd.DataFrame, is_train: bool = True, type: str = "mean"):
    feature_vectors = []
    labels = []
    ids = []

    for video in Path(path).glob('*.npy'):
        id = os.path.basename(video).split('_')[0]
        if id not in df['youtube_id'].values:
            continue
        current_video = np.load(video)
        if len(current_video.shape) <= 1:
            continue
        if type == "mean":
            feature_vectors.append(np.mean(current_video, axis=0))
        elif type == "max":
            feature_vectors.append(np.max(current_video, axis=0))
        ids.append(id)
        if is_train:
            labels.append(df[df['youtube_id'] == id]['label'].values[0])

    feature_vectors = pd.DataFrame(np.vstack(feature_vectors))
    ids = pd.DataFrame({'youtube_id': ids})
    if is_train:
        labels = pd.DataFrame(np.vstack(labels))
        return feature_vectors, labels, ids
    else:
        return feature_vectors, ids

path_train = './Feature_extraction/train/r21d/r2plus1d_18_16_kinetics'
path_val = './Feature_extraction/val/r21d/r2plus1d_18_16_kinetics'
path_test = './Feature_extraction/test/r21d/r2plus1d_18_16_kinetics'

X_train, y_train, ids_train = get_X_y_id(path_train, train_df, type='mean')
X_val, y_val, ids_val = get_X_y_id(path_val, val_df, type='mean')
X_test, ids_test = get_X_y_id(path_test, test_df, False, type='mean')

print(f"Shape de la matriz de training: {X_train.shape}")
print(f"Size de los labels de training: {y_train.shape}")
print(f"Size de los ids de training: {ids_train.shape}\n")
print(f"Shape de la matriz de validation: {X_val.shape}")
print(f"Size de los labels de validation: {y_val.shape}")
print(f"Size de los ids de validation: {ids_val.shape}\n")
print(f"Shape de la matriz de testing: {X_test.shape}")
print(f"Size de los ids de testing: {ids_test.shape}")

def save_df_as_npz(directory: str, filename: str, features: pd.DataFrame, ids, labels=None):
    features = features.to_numpy()
    ids = ids.to_numpy()
    if labels is not None:
        labels = labels.to_numpy()
        np.savez(os.path.join(directory, filename), features=features, ids=ids, labels=labels)
    else:
        np.savez(os.path.join(directory, filename), features=features, ids=ids)

def load_features_ids_labels(filename: str, has_labels=True):
    contents = np.load(filename, allow_pickle=True)
    features = contents['features']
    ids = contents['ids']
    if has_labels:
        labels = contents['labels']
        return features, ids, labels
    return features, ids

os.makedirs('Features', exist_ok=True)
save_df_as_npz(directory='Features', filename='features_train.npz', features=X_train, ids=ids_train, labels=y_train)
save_df_as_npz(directory='Features', filename='features_val.npz', features=X_val, ids=ids_val, labels=y_val)
save_df_as_npz(directory='Features', filename='features_test.npz', features=X_test, ids=ids_test)

X_train_np, ids_train_np, y_train_np = load_features_ids_labels(filename='Features/features_train.npz', has_labels=True)
X_val_np, ids_val_np, y_val_np = load_features_ids_labels(filename='Features/features_val.npz', has_labels=True)
X_test_np, ids_test_np = load_features_ids_labels(filename='Features/features_test.npz', has_labels=False)

# **PREPROCESAMIENTO**
# 1. Eliminar características con varianza muy baja
selector = VarianceThreshold(threshold=0.01)
X_train = selector.fit_transform(X_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

# 2. Aplicar RobustScaler para manejar valores atípicos
robust_scaler = RobustScaler()
X_train = robust_scaler.fit_transform(X_train)
X_val = robust_scaler.transform(X_val)
X_test = robust_scaler.transform(X_test)

# 3. Aplicar StandardScaler después del RobustScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Aplicar PCA para reducir a 50 dimensiones antes de t-SNE y UMAP
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

train_numeric_labels, class_names = pd.factorize(y_train_np.ravel())

def plot_2d(X, labels, title):
    df = pd.DataFrame(X, columns=['Dim 1', 'Dim 2'])
    df['Label'] = labels
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Dim 1', y='Dim 2', hue='Label', palette='viridis', legend='full')
    plt.title(title)
    plt.show()

def plot_3d(X, labels, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()

# Aplicar t-SNE 2D a train, val y test
tsne2 = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
X_train_tsne2 = tsne2.fit_transform(X_train_pca)
X_val_tsne2 = tsne2.fit_transform(X_val_pca)
X_test_tsne2 = tsne2.fit_transform(X_test_pca)
plot_2d(X_train_tsne2, train_numeric_labels, "t-SNE Visualization (2 Components)")

# Aplicar t-SNE 3D a train, val y test
tsne3 = TSNE(n_components=3, random_state=42, max_iter=1000, perplexity=30)
X_train_tsne3 = tsne3.fit_transform(X_train_pca)
X_val_tsne3 = tsne3.fit_transform(X_val_pca)
X_test_tsne3 = tsne3.fit_transform(X_test_pca)
plot_3d(X_train_tsne3, train_numeric_labels, "t-SNE Visualization (3 Components)")

# Aplicar UMAP 2D a train, val y test
umap2 = umap.UMAP(n_components=2, random_state=42)
X_train_umap2 = umap2.fit_transform(X_train_pca)
X_val_umap2 = umap2.fit_transform(X_val_pca)
X_test_umap2 = umap2.fit_transform(X_test_pca)
plot_2d(X_train_umap2, train_numeric_labels, "UMAP Visualization (2 Components)")

# Aplicar UMAP 3D a train, val y test
umap3 = umap.UMAP(n_components=3, random_state=42)
X_train_umap3 = umap3.fit_transform(X_train_pca)
X_val_umap3 = umap3.fit_transform(X_val_pca)
X_test_umap3 = umap3.fit_transform(X_test_pca)
plot_3d(X_train_umap3, train_numeric_labels, "UMAP Visualization (3 Components)")


# Guardar los resultados en la carpeta Reduction_data
os.makedirs('Reduction_data/train', exist_ok=True)
os.makedirs('Reduction_data/test', exist_ok=True)
os.makedirs('Reduction_data/val', exist_ok=True)

np.save('Reduction_data/train/train_tsne_2d.npy', X_train_tsne2)
np.save('Reduction_data/test/test_tsne_2d.npy', X_test_tsne2)
np.save('Reduction_data/val/val_tsne_2d.npy', X_val_tsne2)

np.save('Reduction_data/train/train_tsne_3d.npy', X_train_tsne3)
np.save('Reduction_data/test/test_tsne_3d.npy', X_test_tsne3)
np.save('Reduction_data/val/val_tsne_3d.npy', X_val_tsne3)

np.save('Reduction_data/train/train_umap_2d.npy', X_train_umap2)
np.save('Reduction_data/test/test_umap_2d.npy', X_test_umap2)
np.save('Reduction_data/val/val_umap_2d.npy', X_val_umap2)

np.save('Reduction_data/train/train_umap_3d.npy', X_train_umap3)
np.save('Reduction_data/test/test_umap_3d.npy', X_test_umap3)
np.save('Reduction_data/val/val_umap_3d.npy', X_val_umap3)
