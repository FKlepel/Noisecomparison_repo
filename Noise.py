import numpy as np
np.random.seed(2021)
from matplotlib import pyplot as plt
# import scipy.io
from scipy.io import loadmat
from PIL import Image
import random
import os

# THESE INPUTS ARE GIVEN IN MAIN FILE noise_criteria.py
# but for testing still let here
# most functions from Jaspers files -> rewritten so that file is adaptable to different amount of letters
# important function noise_gen

# input directory of data
# directory = "C:/Users/flore/Documents/Master/ResearchElective/Sciebo/Data/archive/folder"
# os.chdir(directory)
# #which data files will be looked at
# #first 3 signs the same to indicate same participant
# matfiles = ["C01_per1_R_B_VoxBestFit.mat", "C01_per2_R_B_VoxBestFit"]#["S03_per_R_B_VoxBestFit"] #
# imafiles = ["C01_ima1_R_B_VoxBestFit.mat","C01_ima2_R_B_VoxBestFit.mat","C01_ima3_R_B_VoxBestFit.mat","C01_ima4_R_B_VoxBestFit.mat","C01_ima5_R_B_VoxBestFit.mat"]#["S03_ima1_R_B_VoxBestFit.mat","S03_ima2_R_B_VoxBestFit.mat","S03_ima3_R_B_VoxBestFit.mat"]#

# #directory = "C:/Users/flore/Documents/Master/ResearchElective/Noisecomparison/perception"
# #os.chdir(directory)
# #matfiles = ["C01_per1_R_B_VoxBestFit","C01_per2_R_B_VoxBestFit","S01_per_R_B_VoxBestFit", "S02_per_R_B_VoxBestFit", "S03_per_R_B_VoxBestFit","S04_per_R_B_VoxBestFit", "S05_per_R_B_VoxBestFit"]

# #which letteres were used
# letters = ['C','H','S','T']
# #noise_list = ["noiseR1.npy","noiseR2.npy","noiseR1large.npy","noiseR2large.npy","noise80000.npy"]
# noise_list= ["noiseR1.npy"] 

#get the letters into numpy arrays
def norm_it(y):
    x = []
    x = ((y - np.min(y))/np.ptp(y))
    #x = np.stack(x, axis=0)
    return x

def lister(x):
    var = []
    for i in range(len(x)):
        var.append(x[i])
    return var 

def resizo(the_cs, p=28):
    clean_c= []
    
    for i in range(0,len(the_cs)):
        c = []
        c = the_cs[i]
        c = Image.fromarray(c)
        c = c.resize((p,p))
        c = np.asarray(c)
        clean_c.append((c))
    return clean_c
    
#adapt to letter used
def get_letters(arrays,item):
    b = arrays[f"R_{item}"]
    globals()[f"{item}"] = []
    for a in range(0,len(b[1,:])):
        globals()[f"{item}"].append(norm_it(np.transpose(np.reshape(b[:,a],(150,150)))))
    return globals()[f"{item}"]

#adapt to used letters
def get_letter(matfiles, item):
    globals()[f"{item}"] = []
    new_column = []
    
    for i in matfiles:
        arr = loadmat(i)
        globals()[f"{item}".lower()] = get_letters(arr,item)
        globals()[f"{item}"] = globals()[f"{item}"] + globals()[f"{item}".lower()]
        
        new_column = new_column + (np.shape(globals()[f"{item}".lower()])[0] + np.shape(globals()[f"{item}"])[0]) * [i]  
        #globals()[f"{item}"]  = np.append(globals()[f"{item}"], new_column, axis=1)
    return globals()[f"{item}"]
        
#create variations, get ratio
def zoom(the_cs):
    clean_c= []
    zoom_c= []
    x_zeros = np.zeros((10,150))+0.5
   
    y_zeros = np.zeros((170,10))+0.5
      
    for i in range(0,len(the_cs)):
        c = []
        c = the_cs[i]
        clean_c.append((c))
        # adding a 10-thick border of zeros to zoom out then resize, zeros as the average was -8 e-8
        c = np.c_[y_zeros,np.r_[x_zeros,c,x_zeros],y_zeros]
        c = Image.fromarray(c)
        c = c.resize((150,150))
        c = np.asarray(c)
        zoom_c.append((c))
    return clean_c, zoom_c

def normal(the_cs):
    clean_c= []
    zoom_c= []
    for i in range(0,len(the_cs)):
        c = []
        c = the_cs[i]
        clean_c.append((c))
        zoom_c.append((c))
    return clean_c, zoom_c

def angle(the_cs):
    
    clean_c= []
    zoom_c= []
    
    angles = [2, 4, 6, 8, -2, -4, -6, -8]
    
    for a in range(0,len(angles)):
            
        for i in range(0,len(the_cs)):
            c = []
            c = the_cs[i]
            clean_c.append((c))
            # angeling and adding noise
            c = Image.fromarray(c)
            c = c.rotate(angles[a])
            c = np.asarray(c)
            zoom_c.append((c))
            
    return clean_c, zoom_c

#augment tenfold per letter       
def auger(let):
    #del let[32:]

    cleanLs = []
    noisyLs = []
    #cla, nla = normal(let)
    clb, nlb = zoom(let)
    clc, nlc = normal(let)
    cld, nld = angle(let)
    cleanLs = cleanLs + clb + clc + cld
    noisyLs = noisyLs + nlb +nlc + nld
    return cleanLs, noisyLs

def noise_gen(directory, matfiles,imafiles, letters, noise_list):
    # read letters
    for item in letters:
        globals()[f"{item}"] = get_letter(matfiles, item)
        globals()[f"{item}"] = lister(globals()[f"{item}"])    
    # augment letters
    clean = []
    label = []
    dirty = []
    for item in letters:
        globals()[f"c{item.lower()}s"], globals()[f"d{item.lower()}s"] = auger(globals()[f"{item}"])
        globals()[f"label{item}"] = [f"{item}"]*10*len(globals()[f"{item}"])
        clean = clean + globals()[f"c{item.lower()}s"]
        label = label + globals()[f"label{item}"]
        dirty = dirty + globals()[f"d{item.lower()}s"]
    
    #dirty letters = including noise
    #clean letters = without noise
        
    clean = resizo(clean)
    dirty = resizo(dirty)
    dirtraw = dirty
    dirty_placeholder= []
    
    #read in all the noise patterns and attach to the template
    # run through all noise patterns and put out as noise1 to end of noise list with noise(index+1)
    index = 1
    for noise in noise_list:
        noisepat = None
        if noise.split(".")[-1] == "mat":
            noisepat = loadmat(noise)
        else:
            noisepat = np.load(noise)
        
        temp = dirty
        globals()[f"dirty{index}"] = []
        for i in range(len(temp)):
            x = temp[i]+noisepat[i]
            globals()[f"dirty{index}"].append(x)
        globals()[f"dirty{index}"] = norm_it(globals()[f"dirty{index}"])
        dirty_placeholder += lister(globals()[f"dirty{index}"])
        index += 1
    
    dirty = dirty_placeholder
    #dirty = lister(dirty1)+ lister(dirty2) +lister(dirty3)+ lister(dirty4)+lister(dirty5) #+lister(dirtraw)
    clean = clean*len(noise_list)
    label = label*len(noise_list)
    
    #make array with clean perception data, a label and an array with perception+noise
    c_noise = list(zip(clean,label,dirty))
    ima =[]
    labelima = []
    
    #load and resize imagery data, create label
    for item in letters:
        globals()[f"{item}"] = get_letter(imafiles,item)
        globals()[f"label{item}"] = [f"{item}"]*len(globals()[f"{item}"])
        ima = ima + globals()[f"{item}"]
        labelima = labelima + globals()[f"label{item}"]
        
    ima = resizo(ima)
    # imagery files with labels
    c_ima = list(zip(ima, labelima))
    
    ima, labelima = zip(*c_ima)
    
    ima = np.stack(ima, axis=0)
    image_size = ima.shape[1]
    ima = np.reshape(ima, [-1, image_size, image_size, 1])
    ima = ima.astype('float32')
    
    return c_noise, c_ima
