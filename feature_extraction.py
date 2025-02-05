import os
import subprocess
import pandas as pd
from tqdm import tqdm

# Configuración de `video_features`
VIDEO_FEATURES_PATH = "video_features/main.py"
DEVICE = "cuda:0"  # Cambiar a "cpu" si no tienes GPU
FEATURE_TYPE = "r21d"  # Puedes cambiarlo a "i3d", "slowfast" u otro modelo compatible

# Función para extraer características usando `video_features`
def extract_features(csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    # **Cambio: Extraer solo el ID del video (antes del "_")**
    df["id"] = df["path"].apply(lambda x: os.path.basename(x).split("_")[0])

    # Guardar los IDs de video en un archivo temporal
    txt_path = f"{output_folder}/video_ids.txt"
    df["id"].to_csv(txt_path, index=False, header=False)

    # Ejecutar `video_features` mediante subprocess
    command = [
        "python", VIDEO_FEATURES_PATH,
        f"feature_type={FEATURE_TYPE}",
        f"device={DEVICE}",
        f"file_with_video_paths={txt_path}",
        "on_extraction=save_numpy",
        f"output_path={output_folder}"
    ]

    subprocess.run(command, check=True)

# Procesar conjuntos de datos
if __name__ == '__main__':
    extract_features("Data/train_subset_with_paths.csv", "Feature_extraction/train")
    extract_features("Data/test_subset_with_paths.csv", "Feature_extraction/test")
    extract_features("Data/val_subset_with_paths.csv", "Feature_extraction/val")

    print("Extracción de características completada.")
