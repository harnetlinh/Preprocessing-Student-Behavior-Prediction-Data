# merge all csv files in the specified directory
# Usage: python3 <directory>
# Output: merged_<directory>.csv

import os
import sys
import pandas as pd
from datetime import datetime

def merge_csv(directory):
    try:
        files = os.listdir(directory)
        csv_files = [f for f in files if f.endswith('.csv')]
        dfs = []
        for f in csv_files:
            df = pd.read_csv(directory + '/' + f)
            dfs.append(df)
        merged = pd.concat(dfs)
        merged.to_csv(directory + '/merged_' + directory + '.csv', index=False)
    except Exception as e:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_line = str(sys.exc_info()[-1].tb_lineno)
        error_msg = str(e)
        print(current_time + ' Error: ' + error_msg + ' on line: ' + error_line)    
    

if __name__ == '__main__':
    directory = input('Enter the directory: ')
    if not os.path.exists(directory):
        print('Directory does not exist')
        sys.exit(1)
    merge_csv(directory)
