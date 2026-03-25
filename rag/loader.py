import pandas as pd
import os


def load_excel_files(data_path="C-Folder"):
    all_data = []

    for file in os.listdir(data_path):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(data_path, file)
            df = pd.read_excel(file_path)
            df['source_file'] = file
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
