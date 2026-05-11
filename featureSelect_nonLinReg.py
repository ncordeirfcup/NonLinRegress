import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename


import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE

import threading

form = tk.Tk()
form.title("Feature_selection_forNonLinearRegression")
form.geometry("650x300")

tab_parent = ttk.Notebook(form)

tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab2, text="Feature selection")

initialdir=os.getcwd()

def datatr():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)

def ptr():
    global file_pt
    filename_pt = askopenfilename(initialdir=initialdir,title = "Select pre-treated file")
    #NC2E.delete(0, END)
    #NC2E.insert(0, filename1)
    file_pt = pd.read_csv(filename_pt)
    return file_pt


def datats():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)

def lasso(X_uncorr, ytr,nd):
    lasso = Pipeline([('scaler', StandardScaler()),
    ('lasso', LassoCV(alphas=np.logspace(-4, 0, 100), cv=5, random_state=42))])

    lasso.fit(X_uncorr, ytr)

    coef = pd.Series(lasso.named_steps['lasso'].coef_, index=X_uncorr.columns)
    final_descriptors = coef[coef != 0].sort_values().tail(nd).index
    return final_descriptors

def rfs(X_uncorr, ytr, nd):
    importances = []

    for seed in range(50):
        rf = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1)
        rf.fit(X_uncorr, ytr)
        importances.append(rf.feature_importances_)

    mean_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({
       "feature": X_uncorr.columns,
       "importance": mean_importance
    }).sort_values("importance", ascending=False)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rfe = RFE(estimator=rf,
        n_features_to_select=nd,
        step=1)

    # Step 1: explicitly define the subset used for RFE
    top30_features = importance_df["feature"].iloc[:30].tolist()

    # Step 2: fit RFE on this subset
    rfe.fit(X_uncorr[top30_features], ytr)

    # Step 3: extract top 10 correctly
    top10_features = [f for f, s in zip(top30_features, rfe.support_) if s]
    return top10_features


def lsfs(X_uncorr, ytr, nd):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_uncorr)

    lasso = LassoCV(alphas=np.logspace(-4, 0, 100),cv=5,max_iter=5000,random_state=42)
    lasso.fit(X_scaled, ytr)

    coef = pd.Series(lasso.coef_, index=X_uncorr.columns)
    selected_stage1 = coef[coef != 0].index

    X_stage1 = X_uncorr[selected_stage1]
    rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sfs = SequentialFeatureSelector(
    estimator=rf,
    n_features_to_select=nd,
    direction='forward',
    scoring='neg_mean_squared_error',
    cv=cv,
    n_jobs=-1)

    sfs.fit(X_stage1, ytr)
    final_features = X_stage1.columns[sfs.get_support()]
    return final_features

def wait_end(label, tk_var_end, num=0):
    label["text"] = "Processing " + " ." * num
    num += 1
    if num == 4:
        num = 0
    if not tk_var_end.get():
        form.after(500, wait_end, label, tk_var_end, num)


def execute():
    tk_process_lbl = tk.Label(form,font=('Helvetica 12 bold'),fg="blue")
    tk_process_lbl.pack()
    tk_process_lbl.place(x=450,y=265)

    tk_var_end = tk.BooleanVar()
    tk_var_end.set(False)
    wait_end(tk_process_lbl, tk_var_end)
    process = threading.Thread(
        target=process_myFile,
        kwargs=(dict(callback=lambda: tk_var_end.set(True)))
    )
    process.start()

    form.wait_variable(tk_var_end)
    form.after(500, tk_process_lbl.config, dict(text='Process completed'))
    


def process_myFile(callback):
    if NC1.get()=='no':
       Xtr=file1.iloc[:,2:]
       Xts=file2.iloc[:,2:]
       ytr=file1.iloc[:,1:2]
       yts=file2.iloc[:,1:2]
       corrl=float(thirdEntryTabThreer5c2.get())
       var=float(fourthEntryTabThreer5c2.get())
       vt = VarianceThreshold(threshold=var)
    

       X_var_array = vt.fit_transform(Xtr)

       # Get selected column mask
       mask = vt.get_support()

       # Rebuild DataFrame with column names
       X_var = Xtr.loc[:, mask]
       corr = pd.DataFrame(X_var).corr().abs()
       upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
       to_drop = [c for c in upper.columns if any(upper[c] > corrl)]
       X_uncorr = pd.DataFrame(X_var).drop(columns=to_drop)
       X_uncorr.to_csv(str(c_)+'_pt_train_'+str(corrl)+'_'+str(var)+'.csv')
       
    elif NC1.get()=='yes':
       file_pt=ptr()
       Xtr=file1.iloc[:,2:]
       Xts=file2.iloc[:,2:]
       yts=file2.iloc[:,1:2]
       Xpt=file_pt.iloc[:,1:]
       X_uncorr=file1[Xpt.columns]
       ytr=file1.iloc[:,1:2]
       ntr=file1.iloc[:,0:1]  
    nd = int(fifthBoxTabThreer6c2.get())
    if Criterion4.get()=='lasso':
       final_descriptors = lasso(X_uncorr,ytr,nd)
       mn='Lasso_'
    elif Criterion4.get()=='rfs':
       final_descriptors = rfs(X_uncorr,ytr,nd)
       mn='RFselect_'
    elif Criterion4.get()=='lsf':
       final_descriptors = lsfs(X_uncorr,ytr,nd)
       mn='Lasso_SFS_'
    mtr=pd.concat([file1.iloc[:,0:1],ytr,Xtr[final_descriptors]], axis=1)
    mts=pd.concat([file2.iloc[:,0:1],yts,Xts[final_descriptors]], axis=1)
    mtr.to_csv(mn+'selectedDes_tr.csv', index=False)
    mts.to_csv(mn+'selectedDes_ts.csv', index=False)
    callback()

    
def disable_clvar():
    thirdEntryTabThreer5c2['state']='disabled'
    fourthEntryTabThreer5c2['state']='disabled'

def enable_clvar():
    thirdEntryTabThreer5c2['state']='normal'
    fourthEntryTabThreer5c2['state']='normal'       


firstLabelTabThree = tk.Label(tab2, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=95,y=10)
firstEntryTabThree = tk.Entry(tab2, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab2,text='Browse', command=datatr,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab2, text="Select test/screening set",font=("Helvetica", 12))
secondLabelTabThree.place(x=45,y=40)
secondEntryTabThree = tk.Entry(tab2,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab2,text='Browse', command=datats,font=("Helvetica", 10))
b4.place(x=480,y=40)       
       
       
NL1 = tk.Label(tab2, text='Do you have pretreated file: ',font=("Helvetica", 12),anchor=W, justify=LEFT)
NC1 = StringVar() 
NC1.set('no')
NC1_y = tk.Radiobutton(tab2, text='Yes', variable=NC1, value='yes',command=disable_clvar)
NC1_n = tk.Radiobutton(tab2, text='No', variable=NC1, value='no', command=enable_clvar)
NL1.place(x=150,y=75)
NC1_y.place(x=350,y=75)
NC1_n.place(x=420,y=75)

thirdLabelTabThreer2c2=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=200,y=100)
thirdEntryTabThreer5c2=Entry(tab2)
thirdEntryTabThreer5c2.place(x=345,y=100)

fourthLabelTabThreer4c2=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=220,y=125)
fourthEntryTabThreer5c2=Entry(tab2)
fourthEntryTabThreer5c2.place(x=345,y=125)



fifthLabelTabThreer6c2 = Label(tab2, text= 'Maximum descriptors', font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=230,y=160)
fifthBoxTabThreer6c2= Spinbox(tab2, from_=0, to=15, width=5)
fifthBoxTabThreer6c2.place(x=405,y=160)



Criterion_Label4 = ttk.Label(tab2, text="Scoring:", font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4 = StringVar()
Criterion4.set('lasso')
Criterion_acc3 = ttk.Radiobutton(tab2, text='Lasso', variable=Criterion4, value='lasso')
Criterion_roc3 = ttk.Radiobutton(tab2, text='RF selection', variable=Criterion4, value='rfs')
Criterion_roc4 = ttk.Radiobutton(tab2, text='Lasso+SFS', variable=Criterion4, value='lsf')
#Criterion_roc5 = ttk.Radiobutton(tab2, text='NMGD', variable=Criterion4, value='neg_mean_gamma_deviance')
Criterion_Label4.place(x=200,y=195)
Criterion_acc3.place(x=270,y=195)
Criterion_roc3.place(x=340,y=195)
Criterion_roc4.place(x=430,y=195)
#Criterion_roc5.place(x=510,y=285)

b2=Button(tab2, text='Select features', command=execute,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=330,y=225)
    

tab_parent.pack(expand=1, fill='both')

form.mainloop()
