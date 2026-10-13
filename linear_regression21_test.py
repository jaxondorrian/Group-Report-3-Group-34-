# this is important make sure that you have the CVS file in a folder with this file or it wont work
# and make sure you have openpyxl installed  py -m pip install pandas scikit-learn openpyxl

import pandas as pd
from pathlib import Path

folder = Path(__file__).parent
files = list(folder.glob("*.xlsx")) + list(folder.glob("*.csv"))

if not files:
    print("no spreadsheet")
    print("make sure the CVS file is in the same location as the program")
else:
    file = files[0]
    
    if file.suffix == ".csv":
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

  
    print("spreadsheet loaded")
    print("\nFile:", file.name)
    print("\nColumn names:")
    for column in data.columns:
        print("-", column)
      
    print("\nfirst five rows:")
    print(data.head().to_string())




  
