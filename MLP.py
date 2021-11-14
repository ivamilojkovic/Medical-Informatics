import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pickle
import os
import sys

try:
    import pickle
except ModuleNotFoundError as err:
    f = open('C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\fajlovi\\nas_stderr.txt', 'w')
    f.write(err.msg)
    f.close()


def sav2csv(filename):

    outfile = filename[:-4] + '.csv'  
    
    with open(filename, "rb") as f:
        object = pickle.load(f)
        coef = object.coefs_
        
    df = pd.DataFrame(coef)
    df.to_csv(outfile)
 
  
curr_script_path = os.path.realpath(__file__)    
#out_path = os.path.join(curr_script_path, './fajlovi/blbl.txt')
out_path = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\Predictors\\P_MLP.txt'
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

data = pd.read_csv('C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\Data.csv')
X = data.iloc[:,:-1]


fx = open('C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\fajlovi\\X.txt', 'w')
    # fx.write(str(len(list_arg)) + '\n')
    
col_names = X.columns
for i, bi in enumerate(list_arg): 
    if bi == 0:
        name = col_names[i]
        X.drop(name, axis = 1, inplace=True)
    else:
        fx.write(col_names[i] + '\n')
 
fx.write(str(X.shape[1]))    
fx.close()

y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

# Define a model and hyperparameters
model = MLPClassifier()
parameters = {"hidden_layer_sizes":[(10, 2),(10,10,10,10), (10, 5)],  "max_iter": [2000], "alpha": [0.01,0.1,1]}

# Train model
gs = GridSearchCV(model, parameters, cv=3, scoring='f1', verbose=10, n_jobs = -1,refit = True)
gs.fit(X_train, y_train)

# Save the best model for SVM
best_model = gs.best_estimator_
filename = 'C:\\Users\\Phillip Maya\\Desktop\\MI_Project\\PickleFiles\\OW_MLP.pkl'
pickle.dump(best_model, open(filename, 'wb'))

# Covert to csv file
# sav2csv(filename)

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results =[]
results.append(acc)
results.append(pre)
results.append(rec)
results.append(f1)

# Textual file path
path = "C:/Users/Phillip Maya/Desktop/MI_Project/TextFiles" + "MLP.txt"

if os.path.exists(path)==True:
    
    # Delete previos content 
    file = open(path,"r+")
    file.truncate(0)
    file.close()

# Start writing new content
i = 0
while True:
    with open(path,"a") as f:
        f.write(str(round(results[i],4)))
        f.write('\n')
        i += 1
    if i==4:
        break

f.close()


