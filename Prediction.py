# import pandas as pd
# import pickle
# import os

path = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\Predictors\\Predictors_vba.txt'
path_pkl = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\AccessAttachments\\OptimalWeights.pkl'
path_txt = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\AccessAttachments\\UsedPredictors.txt'


try:
    import pickle
    import os
    import pandas as pd
    import numpy as np
except ModuleNotFoundError as err:
    f = open('C:\\Users\\MI_Project\\fajlovi\\nas_stderr.txt', 'w')
    f.write(err.msg)
    f.close()


X = pd.read_csv(path, delimiter='/', header=None)
X.drop(X.columns[-1], axis = 1, inplace = True)

with open(path_pkl, "rb") as f:
    model = pickle.load(f)
    
file_used = open(path_txt, 'r')
lines = file_used.readlines()

temp = []
for i, line in enumerate(lines):
    if int(line)==1:
        temp.append(X.columns[i])
        # X.drop(X.columns[i], axis = 1, inplace = True)

X = pd.DataFrame(np.array(temp).reshape(1,-1))        
result = model.predict_proba(X)

out_path = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\TextFiles\\Result.txt'
file = open(out_path, 'w')
file.write(str(result[0,1]))
file.close()

file_used.close()
        
os.remove(path_pkl)
os.remove(path_txt)
