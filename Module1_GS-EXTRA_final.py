import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from cross_validation2 import cross_validation as cv2
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from rm2 import rm2
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')
import pickle
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import sys
from catboost import CatBoostRegressor


initialdir=os.getcwd()
#RF=RandomForestClassifier()
sc = StandardScaler()

def write_versions(filer):
    import sklearn
    import xgboost
    import lightgbm
    filer.write('Scikit-learn version: '+ str(sklearn.__version__)+'\n')
    filer.write('XGBoost version: '+ str(xgboost.__version__)+'\n')
    filer.write('LightGBM version: '+ str(lightgbm.__version__)+'\n')
    filer.write('Numpy version: '+str(np.__version__)+'\n')
    filer.write('Pandas version: '+str(pd.__version__)+'\n')
    filer.write('Python version:' + str(sys.version))
    filer.write('\n')



def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))
    
def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)

def RMSE(df,logHC,Pred_loo):
    #df=df.loc[~df[str(logHC)].isna(),:]
    df['Active']=df[str(logHC)]
    df['Predict']=df[str(Pred_loo)]
    df['diff']=(df['Active']-df['Predict'])**2
    rmse=np.sqrt((df['diff']).sum()/(df.shape[0]))
    return rmse
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select parameter file")
    thirdEntryTabThree_x.delete(0, END)
    thirdEntryTabThree_x.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)
    
def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > cthreshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                    del X[colname] # deleting the column from the dataset
    return X   

def variance(X,threshold):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X

def aardpr(df):
    df.columns=['Active','Predict']
    #df['Active2']=np.exp(df['Active'])
    #df['Predict2']= np.exp(df['Predict'])
    df['diff']=abs(df['Active']-df['Predict'])
    aard=(100*(df['diff']/df['Active'])).sum()/(df.shape[0])
    return aard

def parse_value(v):
    v = str(v).strip()
    print(v)

    # Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # None
    if v.lower() == "none":
        return None

    # Integer
    try:
        if "." in v:
            if v.split('.')[1]=='0':
                if int(v.split('.')[0])>0:
                   return int(v.split('.')[0])
            elif float(v)>0:
                   return float(v)
        else: 
            return int(v)
                    
    except:
        pass

    # Float

    # String
    return v


def csv_to_param_grid(csv_file):
    df = pd.read_csv(csv_file)

    param_grid = {}

    for col in df.columns:
        values = df[col].dropna().unique()
        param_grid[col] = [parse_value(v) for v in values] 

    return param_grid
    

def selected():
    param_grid=csv_to_param_grid(filename3)
    print(param_grid)
    print(random_state_entry.get())
    rd=int(random_state_entry.get())
    if Criterion.get()==1:
        estimator1=RandomForestRegressor(random_state=rd)
        rn='RF'
    elif Criterion.get()==2:
        estimator1=KNeighborsRegressor()
        rn='KNN'       
    elif Criterion.get()==3:
         estimator1 = AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=rd), random_state=rd)
         rn='AB'

    elif Criterion.get()==4:
        estimator1=SVR()
        rn='SVR'        
    elif Criterion.get()==5:
         estimator1=GradientBoostingRegressor(verbose=0, random_state=rd)
         rn='GB'       
    elif Criterion.get()==6:
         estimator1=MLPRegressor(max_iter=7000, random_state=rd)
         rn='MLP'
         #param_grid ['hidden_layer_sizes']=[(50,50)]
         #param_grid= {"hidden_layer_sizes": [(50, 50)], "activation": ["identity", "logistic", "tanh", "relu"],
                      #'alpha': [0.0001, 0.001, 0.01, 0.1],'learning_rate': ['constant','adaptive', 'invscaling']}
         if int(thirdEntryTabThreer3c1_h.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),)]
         elif 'hidden_layer_sizes' in file3.columns:
               lst = file3['hidden_layer_sizes'].values.tolist()
               result = [tuple(map(int, x.split(','))) for x in lst]
               param_grid['hidden_layer_sizes']=result
         else:
               pass
         if int(thirdEntryTabThreer3c1_h1.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()))]
         if int(thirdEntryTabThreer3c1_h2.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()),int(thirdEntryTabThreer3c1_h2.get()))]
         print(param_grid)
    elif Criterion.get()==7:
         estimator1=XGBRegressor(objective="reg:squarederror",random_state=rd)
         rn='XGB'
    elif Criterion.get()==8:
         estimator1=ExtraTreesRegressor(random_state=rd)
         rn='ETR'
    elif Criterion.get()==9:
         estimator1=LGBMRegressor(objective='regression',random_state=rd)
         rn='LGB' 
    elif Criterion.get()==10:
         estimator1=DecisionTreeRegressor(random_state=rd)
         rn='DTR'
    elif Criterion.get()==11:
         estimator1=CatBoostRegressor(random_state=rd)
         rn='CBR'  
    else:
        pass
    rn='g_'+rn
    return estimator1,param_grid,rn

   
 
def sol():
    X_train=file1.iloc[:,2:].round(decimals=4)
    ytr=file1.iloc[:,1:2].round(decimals=4)
    ntr=file1.iloc[:,0:1]
    if stc1.get()=='yes':
       Xtr1 = pd.DataFrame(np.round(sc.fit_transform(X_train),4))
       Xtr1.columns=X_train.columns
    elif stc1.get()=='no':
       Xtr1=pd.DataFrame(X_train)
    estimator,param_grid,rn=selected()
    cvg=thirdEntryTabOne.get()
    cvg=int(cvg)
    cv1 = KFold(n_splits=cvg, shuffle=True, random_state=42)
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    #param_grid=paramgrid()
    clf = GridSearchCV(cv=cv1, estimator=estimator, param_grid=param_grid, n_jobs=-1, verbose=1)
    print(clf)
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    if cthreshold<1 or vthreshold>0:
       print('With Pretreatment')
       Xtr=pretreat(Xtr1,cthreshold,vthreshold)
       pd.DataFrame(Xtr).to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
       #pd.DataFrame(Xts).to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    else:
       print('Without Pretreatment')
       Xtr=pd.DataFrame(Xtr1)
       Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    clf.fit(np.array(Xtr),ytr)
    global clfb
    clfb=clf.best_estimator_
    pname='best_model_'+str(rn)+'.pkl'
    pickle.dump(clfb, open(pname, 'wb'))
    print(clfb)
    clfb.fit(Xtr,ytr)
    
    
    filer = open(str(c_)+rn+"_tr.txt","w")
    write_versions(filer)
    filer.write('The best estimator is: '+'\n')
    filer.write(str(clfb))
    filer.write("\n")
    filer.write('Dependent parameter: '+str(ytr.columns[0])+'\n')
    filer.write('Number of training set compounds: '+str(Xtr.shape[0])+'\n')
    #filer.write('Number of test set compounds: '+str(Xts.shape[0])+'\n')
    filer.write(str(cvm)+' fold cross validation statistics are: '+'\n')
    filer.write('\n')
    Xtr=pd.DataFrame(Xtr)
    Xtr.columns=X_train.columns.tolist()
    writefile2(Xtr,ytr,ntr,clfb,cvm,filer,rn)
    filer.write("\n")
   

def writefile2(Xtr,ytr,ntr,model,cvm,filer,rn):
    cvv=cv2(Xtr,ytr,ntr,model,cvm)
    r2,mae,q2lmo,rm2tr,drm2tr,ls,aard=cvv.fit()
    print(ls.shape)
    dftr1=pd.concat([ntr,Xtr],axis=1)
     
    #dftr2=ls.iloc[:,0:2]
    dftr=pd.merge(dftr1,ls.iloc[:,0:3],on=ls.iloc[:,0:1].columns[0],how='left')
    dftr.to_csv(str(c_)+str(rn)+"_trpr.csv",index=False)
    filer.write('R2: '+str(r2)+"\n")
    filer.write(str(cvm)+'-fold cross-validated R2: '+str(q2lmo)+"\n")
    filer.write('Mean absolute error: '+str(mae)+"\n")
    filer.write('Rm2tr '+str(rm2tr)+"\n")
    filer.write('Delta Rm2tr '+str(drm2tr)+"\n")
    filer.write('AARDcv: '+str(aard)+"\n")
    if ytr.columns[0] in file2.columns:
       Xts=file2[Xtr.columns].round(decimals=4)
       #Xts==file2[X_train.columns]
       if stc1.get()=='yes':
          Xts = np.round(sc.transform(Xts),4)
       elif stc1.get()=='no':
          Xts=Xts
       nts=file2.iloc[:,0:1]
       yts=file2.iloc[:,1:2].round(decimals=4)
       ytspr=pd.DataFrame(model.predict(Xts))
       Xts=pd.DataFrame(Xts)
       Xts.columns=Xtr.columns
       ytspr.columns=['Pred']
       dtspr=pd.concat([yts,ytspr],axis=1)
       AARDts=aardpr(dtspr)
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
       tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
       tsdf.columns=['Active','Predict']
       #aardts=aard(tsdf)
       tsdf['Aver']=ytr.values.mean()
       tsdf['Aver2']=tsdf['Predict'].mean()
       tsdf['diff']=tsdf['Active']-tsdf['Predict']
       tsdf['diff2']=tsdf['Active']-tsdf['Aver']
       tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
       maets=mean_absolute_error(tsdf['Active'],tsdf['Predict'])
       r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
       r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
       RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
       dfts=pd.concat([nts,Xts,yts,ytspr],axis=1)
       dfts.to_csv(str(c_)+str(rn)+"_tspr.csv",index=False)
       filer.write("\n")
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations: '+str(yts.shape[0])+"\n")
       filer.write('MAEtest: '+ str(maets)+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write('AARD_test: '+str(AARDts)+"\n")
       filer.write("\n")
       
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(model.predict(Xts))
        ytspr.columns=['Pred']
        #adts=apdom(Xts[a],Xtr)
        #yadts=adts.fit()
        dfts=pd.concat([nts,Xts,ytspr],axis=1)
        dfts.to_csv(str(c_)+str(rn)+"_scpr.csv",index=False)
    
# ==========================
# Improved UI Section Only
# ==========================

form = tk.Tk()
form.title("NonLinRegress")
form.geometry("850x750")
form.configure(bg="#f4f6f9")
form.resizable(False, False)

# ---------- Style ----------
style = ttk.Style()
style.theme_use("clam")

style.configure("TNotebook", background="#f4f6f9", borderwidth=0)
style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[15, 8])
style.configure("TLabel", background="#f4f6f9", font=("Segoe UI", 11))
style.configure("TButton",
                font=("Segoe UI", 10, "bold"),
                padding=6)

style.configure("Accent.TButton",
                font=("Segoe UI", 11, "bold"),
                foreground="white",
                background="#1f77b4")

# ---------- Notebook ----------
tab_parent = ttk.Notebook(form)
tab_parent.pack(expand=1, fill="both", padx=15, pady=15)

tab1 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text=" Grid Search Non-Linear Model ")

# ---------- Frames ----------
file_frame = ttk.LabelFrame(tab1, text=" Dataset & Parameters ", padding=10)
file_frame.pack(fill="x", padx=10, pady=10)

method_frame = ttk.LabelFrame(tab1, text=" Machine Learning Method ", padding=10)
method_frame.pack(fill="x", padx=10, pady=10)

settings_frame = ttk.LabelFrame(tab1, text=" Model Settings ", padding=10)
settings_frame.pack(fill="x", padx=10, pady=10)

# ===============================
# FILE SELECTION SECTION
# ===============================

ttk.Label(file_frame, text="Sub-training set:").grid(row=0, column=0, sticky="w", pady=5)
firstEntryTabThree = ttk.Entry(file_frame, width=60)
firstEntryTabThree.grid(row=0, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data1).grid(row=0, column=2)

ttk.Label(file_frame, text="Test set:").grid(row=1, column=0, sticky="w", pady=5)
secondEntryTabThree = ttk.Entry(file_frame, width=60)
secondEntryTabThree.grid(row=1, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data2).grid(row=1, column=2)

ttk.Label(file_frame, text="Parameter file:").grid(row=2, column=0, sticky="w", pady=5)
thirdEntryTabThree_x = ttk.Entry(file_frame, width=60)
thirdEntryTabThree_x.grid(row=2, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data3).grid(row=2, column=2)

# ===============================
# STANDARDIZATION
# ===============================

ttk.Label(file_frame, text="Standardization:").grid(row=3, column=0, sticky="w", pady=10)

stc1 = StringVar()
ttk.Radiobutton(file_frame, text='Yes', variable=stc1, value='yes').grid(row=3, column=1, sticky="w")
ttk.Radiobutton(file_frame, text='No', variable=stc1, value='no').grid(row=3, column=1, padx=60, sticky="w")

# ===============================
# METHOD SECTION
# ===============================

Criterion = IntVar()

methods = [
    ("Random Forest", 1),
    ("k-Nearest Neighbors", 2),
    ("Support Vector Machine", 4),
    ("Gradient Boosting", 5),
    ("Extreme Gradient Boosting", 7),
    ("Extra Trees", 8),
    ("LightGBM", 9),
    ("Decision Tree", 10),
    ("Multilayer Perceptron", 6),
    ("Adaboost", 3),
    ("Catboost",11)
]

row = 0
col = 0
for text, val in methods:
    ttk.Radiobutton(method_frame, text=text,
                    variable=Criterion,
                    value=val,
                    command=selected).grid(row=row, column=col, padx=10, pady=5, sticky="w")
    col += 1
    if col == 3:
        col = 0
        row += 1

# ===============================
# MLP Hidden Layers (Improved Alignment)
# ===============================

ttk.Label(method_frame, text="Hidden Layers (MLP only and not provided in parameter file):").grid(
    row=row+1, column=0, pady=10, sticky="w"
)

# Inner frame to keep entries close together
hidden_frame = ttk.Frame(method_frame)
hidden_frame.grid(row=row+1, column=1, columnspan=3, sticky="w")

v1 = IntVar(value=0)
v2 = IntVar(value=0)
v3 = IntVar(value=0)

thirdEntryTabThreer3c1_h = ttk.Entry(hidden_frame, textvariable=v1, width=6)
thirdEntryTabThreer3c1_h.pack(side="left", padx=(0, 5))

ttk.Label(hidden_frame, text=",").pack(side="left")

thirdEntryTabThreer3c1_h1 = ttk.Entry(hidden_frame, textvariable=v2, width=6)
thirdEntryTabThreer3c1_h1.pack(side="left", padx=5)

ttk.Label(hidden_frame, text=",").pack(side="left")

thirdEntryTabThreer3c1_h2 = ttk.Entry(hidden_frame, textvariable=v3, width=6)
thirdEntryTabThreer3c1_h2.pack(side="left", padx=(5, 0))# ===============================

# ===============================
# MODEL SETTINGS (COMPACT - 2 COLUMN)
# ===============================

ttk.Label(settings_frame, text="Correlation cut-off:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
thirdEntryTabThreer3c1 = ttk.Entry(settings_frame, width=12)
thirdEntryTabThreer3c1.grid(row=0, column=1, padx=5)

ttk.Label(settings_frame, text="Variance cut-off:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
fourthEntryTabThreer5c1 = ttk.Entry(settings_frame, width=12)
fourthEntryTabThreer5c1.grid(row=0, column=3, padx=5)

ttk.Label(settings_frame, text="CV (Grid Search):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
thirdEntryTabOne = ttk.Entry(settings_frame, width=12)
thirdEntryTabOne.grid(row=1, column=1, padx=5)

ttk.Label(settings_frame, text="CV (Predictability):").grid(row=1, column=2, padx=5, pady=2, sticky="w")
forthEntryTabOne = ttk.Entry(settings_frame, width=12)
forthEntryTabOne.grid(row=1, column=3, padx=5)

# ===============================
# RANDOM STATE (NEW SECTION)
# ===============================

ttk.Label(method_frame, text="Random State:").grid(
    row=row+2, column=0, pady=10, sticky="w"
)

random_state_var = IntVar(value=42)  # default value

random_state_entry = ttk.Entry(method_frame, textvariable=random_state_var, width=10)
random_state_entry.grid(row=row+2, column=1, sticky="w")

# ===============================
# GENERATE BUTTON
# ===============================

generate_btn = ttk.Button(
    tab1,
    text=" Generate Model ",
    command=sol,
    style="Accent.TButton"
)
generate_btn.pack(pady=5)

form.mainloop()
