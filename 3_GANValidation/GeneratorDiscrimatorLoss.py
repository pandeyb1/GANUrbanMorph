import os
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd

os.chdir("/Users/9oy/Documents/Projects/IM3/EvaluationP/")

lr =  [0.0001,0.0002,0.0005,0.001,0.005]

GLfull = []
DLfull = []
for i in lr:

    fnameG = f"LULCCond_BFgenerator_withLatentVector_100L_LR_{i}_GLOSS.csv"
    fnameD = f"LULCCond_BFgenerator_withLatentVector_100L_LR_{i}_DLOSS.csv"
    GL= pd.read_csv(fnameG,header=None).reset_index()
    DL= pd.read_csv(fnameD,header=None).reset_index()
    GL.columns = ["Epoch","Loss"]
    DL.columns = ["Epoch","Loss"]
    GL["LR"] = i
    DL["LR"] = i
    GLfull.append(GL)
    DLfull.append(DL)

GLfull = pd.concat(GLfull)
DLfull = pd.concat(DLfull)
GLfullfinal = GLfull.loc[GLfull.Epoch==499]
DLfullfinal = DLfull.loc[DLfull.Epoch==499]

unique_lrs = GLfull['LR'].unique()
alphas = [0.9,0.9,0.8,0.7,0.25]
plt.figure()
for i in range(len(unique_lrs)):
    lr = unique_lrs[i]
    subset = GLfull[GLfull['LR'] == lr]
    subset = subset[subset["Epoch"] <=999]
    plt.plot(subset['Epoch'], subset['Loss'], label=f'LR = {lr}',alpha=alphas[i])
plt.xlabel('Epoch')
plt.ylabel('Average Generator Loss')
plt.legend()
plt.title("Building Footprint Fraction Generator")
plt.show()
print(GLfullfinal)

GLfull = []
DLfull = []
for i in lr:

    fnameG = f"BFCond_BHgenerator_withLatentVector_100L_LR_{i}_GLOSS.csv"
    fnameD = f"BFCond_BHgenerator_withLatentVector_100L_LR_{i}_DLOSS.csv"
    GL= pd.read_csv(fnameG).reset_index()
    DL= pd.read_csv(fnameD).reset_index()
    GL.columns = ["Epoch","Loss"]
    DL.columns = ["Epoch","Loss"]
    GL["LR"] = i
    DL["LR"] = i
    GLfull.append(GL)
    DLfull.append(DL)

GLfull = pd.concat(GLfull)
DLfull = pd.concat(DLfull)
GLfullfinal = GLfull.loc[GLfull.Epoch==499]
DLfullfinal = DLfull.loc[DLfull.Epoch==499]

unique_lrs = GLfull['LR'].unique()
plt.figure()
for i in range(len(unique_lrs)):
    lr = unique_lrs[i]
    subset = GLfull[GLfull['LR'] == lr]
    subset = subset[subset["Epoch"] <=999]
    plt.plot(subset['Epoch'], subset['Loss'], label=f'LR = {lr}',alpha=alphas[i])
plt.xlabel('Epoch')
plt.ylabel('Average Generator Loss')
plt.legend()
plt.title("Average Building Heights Generator")
plt.ylim(8,16)
plt.show()
print(GLfullfinal)