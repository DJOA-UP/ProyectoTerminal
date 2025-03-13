import pdfplumber
import pandas as pd

path = r'Direccion del archivo PDF'

with pdfplumber.open(path) as pdf:
    table_data = []
    for page in pdf.pages:
        tables = page.extract_table()
        if tables:
            table_data.extend(tables)

df = pd.DataFrame(table_data[1:], columns=table_data[0])  
print(df)

for column in df.columns:
    column_df = df[[column]]
    column_csv_path = f'direccion de carpeta de archivo PDF\\{column}.csv'
    column_df.to_csv(column_csv_path, index=False)
    print(f"Saved {column} to {column_csv_path}")