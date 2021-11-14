import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_csv('C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\cardiovascular_heart_disease-datasets\\S1Data.csv')
new_data = data.iloc[:,7:]
scaler = MinMaxScaler()
scaler.fit(new_data)

# curr_script_path = os.path.realpath(__file__)    
out_path = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\Predictors\\Predictors_vba.txt'
fajl = open(out_path, 'w')

list_arg = []

if len(sys.argv) == 0:
    fajl.write("NEMA ARGUMENATA\n\n")
else:
    # fajl.write(f"IMA IH {len(sys.argv)} \n\n")
    for i, it in enumerate(sys.argv):
        # fajl.write(it + '\n')
        if i != 0:
            fajl.write(it + '\n')
            list_arg.append(int(it))
           
fajl.close()    


bin_list = np.array(list_arg[:5]).reshape(1,-1)
num_list = np.array(list_arg[5:]).reshape(1, -1)

new_num_list = scaler.transform(num_list)

new_list = np.concatenate((bin_list, new_num_list), axis=1)

out_path = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\Predictors\\Predictors_vba.txt'
fajl = open(out_path, 'w')

for i in range(0, new_list.shape[1]):
    fajl.write(str(new_list[0,i]) + "/")
    
fajl.close()