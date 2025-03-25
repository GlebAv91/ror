import pandas as pd
import os

os.chdir("D:/Projects/Ror/ror")
print("Current Working Directory:", os.getcwd())

# Path to your file
file_path = "input/input_patient_1.xlsx"

# Load the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Display basic info about the data
print(data.info())
