import pandas as pd
import os

def load_excel_files(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(folder_path, file)

            excel_data = pd.read_excel(file_path, sheet_name=None)

            for sheet_name, df in excel_data.items():
                df = df.fillna("")

                for index, row in df.iterrows():
                    row_text = f"File: {file} | Sheet: {sheet_name} | Row {index} | "

                    for col in df.columns:
                        row_text += f"{col}: {row[col]} | "

                    documents.append(row_text)

    return documents
