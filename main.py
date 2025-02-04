import pandas as pd
import os

# Directorios de los archivos existentes
data_folder = "Data"
paths_folder = "Paths"

# Archivos CSV con los IDs y etiquetas
csv_files = {
    "test": os.path.join(data_folder, "test_subset.csv"),
    "train": os.path.join(data_folder, "train_subset.csv"),
    "val": os.path.join(data_folder, "val_subset.csv")
}

# Archivos TXT con los paths de los videos
txt_files = {
    "test": os.path.join(paths_folder, "test.txt"),
    "train": os.path.join(paths_folder, "train.txt"),
    "val": os.path.join(paths_folder, "val.txt")
}


# Función para cargar los archivos TXT y mapear youtube_id a path
def load_video_paths(txt_file):
    video_paths = {}
    with open(txt_file, "r") as f:
        for line in f:
            path = line.strip()
            filename = os.path.basename(path)
            youtube_id = filename.split("_")[0]  # Extraer ID antes del primer "_"
            video_paths[youtube_id] = path
    return video_paths


# Procesar cada conjunto (test, train, val)
for subset, csv_file in csv_files.items():
    df = pd.read_csv(csv_file)
    video_paths = load_video_paths(txt_files[subset])

    # Agregar la columna de path
    df["path"] = df["youtube_id"].map(video_paths)

    # Guardar el nuevo CSV con paths incluidos
    output_csv = os.path.join(data_folder, f"{subset}_subset_with_paths.csv")
    df.to_csv(output_csv, index=False)
    print(f"Archivo generado: {output_csv}")

# Código original comentado (creación de archivos y conteo de videos por carpeta)
# def count_videos_in_folder(folder_path):
#     video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.mpeg', '*.mpg')
#     video_count = 0
#     video_files = []  # Lista para almacenar las rutas completas de los archivos
#
#     if not os.path.exists(folder_path):
#         print(f"La carpeta {folder_path} no existe.")
#         return 0, []
#
#     for root, _, files in os.walk(folder_path):
#         for ext in video_extensions:
#             matched_files = fnmatch.filter(files, ext)
#             video_count += len(matched_files)
#             # Agregar las rutas completas de los archivos encontrados a la lista
#             for file in matched_files:
#                 video_files.append(os.path.join(root, file))
#
#     return video_count, video_files
#
# def save_paths_to_file(file_name, video_files):
#     with open(file_name, 'w') as file:
#         for video in video_files:
#             file.write(video + '\n')
#
# folders = [
#     r"G:\.shortcut-targets-by-id\1lC0CxE7lfODa-bWinGyEbB-4aE3HDcuv",  # test folder
#     r"G:\.shortcut-targets-by-id\1n5yaFEuSp2yBTSRo7uPhTRPAPXl-Iy74",  # train folder
#     r"G:\.shortcut-targets-by-id\1JnLhj1MZO_sNZS02y9Fn-lOgQv4S7r17"  # val folder
# ]
#
# output_files = ['test.txt', 'train.txt', 'val.txt']
#
# for index, (folder, output_file) in enumerate(zip(folders, output_files)):
#     count, video_files = count_videos_in_folder(folder)
#     print(f"\nLa carpeta {folder} contiene {count} videos.")
#
#     save_paths_to_file(output_file, video_files)
#     print(f"Las rutas se han guardado en {output_file}.")
