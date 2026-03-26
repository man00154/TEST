import pandas as pd
import os


def load_excel_files(data_path="C-Folder"):
    all_data = []

    for file in os.listdir(data_path):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(data_path, file), engine="openpyxl")

        elif file.endswith(".xls"):
            df = pd.read_excel(os.path.join(data_path, file), engine="xlrd")

        else:
            continue

        df["source_file"] = file
        all_data.append(df)

    if not all_data:
        raise ValueError("No valid Excel files found")

    return pd.concat(all_data, ignore_index=True)
