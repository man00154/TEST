import pandas as pd
import os

def load_excel_files(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(folder_path, file)

            try:
                excel_data = pd.read_excel(file_path, sheet_name=None)

                for sheet_name, df in excel_data.items():
                    df = df.fillna("")

                    for index, row in df.iterrows():
                        row_text = f"""
                        File: {file}
                        Sheet: {sheet_name}
                        Row: {index}
                        Data: {row.to_dict()}
                        """
                        documents.append(row_text)

            except Exception as e:
                print(f"Error reading {file}: {e}")

    return documents
