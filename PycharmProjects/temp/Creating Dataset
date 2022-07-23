import pandas as pd
import os


path = 'filtered data/'

full_data = pd.DataFrame()

for entry in sorted(os.listdir(path)):
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.txt'):
            data = pd.read_csv(path + entry, sep=' ', header=None)
            data.drop([129, 130], inplace=True, axis=1)
            data['classs'] = entry[-10:-8]
            full_data = pd.concat([full_data, data], ignore_index=True)

print (full_data.shape)

print (full_data.info())