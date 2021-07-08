from PIL import ImageChops, Image
import math, operator
from Noise import noise_gen
import numpy as np
from matplotlib import pyplot
from skimage.metrics import structural_similarity as ssim
import cv2
import argparse
from astropy.table import QTable
from numpy import *
from scipy.spatial.distance import pdist, squareform
import pandas as pd
#from pandas import *
import researchpy as rp
import scipy.stats as stats
import os
import sys

#noise80000 = biologically inspired noise
#noiseR2large = HRF noise, different masking
#noiseR1large = HRF noise, different masking
#noiseR2 = HRF noise, different subject
#noiseR1 = HRF noise

#input directory of data
directory = "C:/Users/flore/Documents/Master/ResearchElective/Sciebo/Data/archive/folder"
os.chdir(directory)
#which data files will be looked at
#first 3 signs the same to indicate same participant
#matfiles = ["C01_per1_R_B_VoxBestFit.mat", "C01_per2_R_B_VoxBestFit"]#["S03_per_R_B_VoxBestFit"] #
#imafiles = ["C01_ima1_R_B_VoxBestFit.mat","C01_ima2_R_B_VoxBestFit.mat","C01_ima3_R_B_VoxBestFit.mat","C01_ima4_R_B_VoxBestFit.mat","C01_ima5_R_B_VoxBestFit.mat"]#["S03_ima1_R_B_VoxBestFit.mat","S03_ima2_R_B_VoxBestFit.mat","S03_ima3_R_B_VoxBestFit.mat"]#

matfiles1 =["S03_per_R_B_VoxBestFit"]
imafiles1 =["S03_ima1_R_B_VoxBestFit.mat","S03_ima2_R_B_VoxBestFit.mat","S03_ima3_R_B_VoxBestFit.mat", "S03_ima4_R_B_VoxBestFit.mat"]

matfiles2 =["S05_per_R_B_VoxBestFit"]
imafiles2 =["S05_ima1_R_B_VoxBestFit.mat","S05_ima2_R_B_VoxBestFit.mat","S05_ima3_R_B_VoxBestFit.mat"]

matfiles3 =["times_perception_1_270421_2057_retrain", "times_perception_2_270421_2203_retrain"]
imafiles3 =["times_rt_imagery_1_270421_2105_retrain","times_rt_imagery_2_270421_2112_retrain","times_rt_imagery_3_270421_2119_retrain", "times_rt_imagery_4_270421_2225_retrain"]

#directory = "C:/Users/flore/Documents/Master/ResearchElective/Noisecomparison/perception"
#os.chdir(directory)
#matfiles = ["C01_per1_R_B_VoxBestFit","C01_per2_R_B_VoxBestFit","S01_per_R_B_VoxBestFit", "S02_per_R_B_VoxBestFit", "S03_per_R_B_VoxBestFit","S04_per_R_B_VoxBestFit", "S05_per_R_B_VoxBestFit"]

#which letters were used
#only use letters that were used for all participants
letters = ['C','S','T']
noise_list = ["noiseR1.npy","noiseR2.npy","noiseR1large.npy","noiseR2large.npy","noise80000.npy"]
#noise_list= ["noise80000.npy"]

#used measures for comparisons
measures = ['MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean']

# to include a certain subject the matfile name plus the imafile name need to be included here in a list item
# in case only one person is in here, no average analysis over all participants will be done
# always first 3 signs of the first matfile used as identifier
sub_list = [[matfiles1,imafiles1],  [matfiles2, imafiles2], [matfiles3,imafiles3]] #, [matfiles3,imafiles3]

#in case only one run is supposed to be analysed, then just put only one run in the matfiles/imafiles of the specific participant and save data after

#########################

def calculate_psnr(img1, img2, max_value=28*28):
    #Calculating peak signal-to-noise ratio (PSNR) between two images.
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
	
    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error

def compare(imageA, imageB):
    # Calculate the MSE and SSIM
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # Return the SSIM. The higher the value, the more "similar" the two images are.
    return s


#dcov, dvar,cent_dist_dcor to calculate distance correlation between two matrices https://en.wikipedia.org/wiki/Distance_correlation
def dcov(X, Y):
    #Computes the distance covariance between matrices X and Y.
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov

def dvar(X):
    #Computes the distance variance of a matrix X
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))

def cent_dist(X):
    #Computes the pairwise euclidean distance between rows of X and centers
    #each cell of the distance matrix with row mean, column mean, and grand mean
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM

def dcor(X, Y):
    #Computes the distance correlation between two matrices X and Y.
    #X and Y must have the same number of rows.
    #>>> X = np.matrix('1;2;3;4;5')
    #>>> Y = np.matrix('1;2;9;4;4')
    #>>> dcor(X, Y)
    #0.76267624241686649
    assert X.shape[0] == Y.shape[0]
    A = cent_dist(X)
    B = cent_dist(Y)
    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)
    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)
    return dcor

if __name__ == "__main__":
    #prepare to put all participants that run through the analysis in one big file and make analysis with that one
    #make overall analysis possible
    # every thing with a _long extension is used for overall analysis 
    ima_data_long = []
    dirty_data_long = []
    for noise in range(len(noise_list)):
        globals()[f"dirty{noise+1}_data_long"] = []        
        for item in letters:
            globals()[f"dirty{noise+1}_data_{item}_long"] = []
            globals()[f"ima_data_{item}_long"] = []
    
    #here starts analysis per person, plots labeled with 3 first items of file name in matfiles
    for matfiles,imafiles in sub_list:
        #if this file is supposed to be used to minimise error at some point sys.argv can be used to give python file the noise_list that is supposed to be used        
        noise_list = sys.argv[-1] if len(sys.argv) > 2 else noise_list
        #for noise_gen see Noise.py, in general: here perception files and imagery files are read, resized, augmented, normalised and then the noise is added to the perception files
        #c_noise are perception images with and without noise, c_ima are only imagery images
        c_noise, c_ima = noise_gen(directory, matfiles,imafiles, letters, noise_list)
        #empty array to fill with data
        dirty_data = []
        dirty_data_ima = []
        #empty arrays for each letter and each kind of noise
        for item in letters:
            for noise in range(len(noise_list)):
                globals()[f"{item}_list_{noise+1}_noiseperc"] = []
            globals()[f"{item}_list"] = []
            globals()[f"{item}_list_ima"] = []
            globals()[f"{item}_list_imamean"] = []
            globals()[f"dirty_data_{item}_ima"] = []
            globals()[f"data_{item}_ima_vs_av"] = []
        
        #compare noisy to clean perception data
        for frame in range(len(c_noise)):
            #indicate two pictures, one with noise, complement without noise
            image1 = c_noise[frame][0]
            image2 = c_noise[frame][2]
            gray1 = image1
            gray2 = image2
            # Check for same size and ratio and report accordingly
            ho, wo = image1.shape
            hc, wc = image2.shape
            ratio_orig = ho/wo
            ratio_comp = hc/wc
            dim = (wc, hc)
            if round(ratio_orig, 2) != round(ratio_comp, 2):
             print("\nImages not of the same dimension. Check input.")
             exit()
            
            # Resize first image if the second image is smaller
            elif ho > hc and wo > wc:
             print("\nResizing original image for analysis...")
             gray1 = cv2.resize(gray1, dim)
            
            elif ho < hc and wo < wc:
             print("\nCompressed image has a larger dimension than the original. Check input.")
             exit()
            
            if round(ratio_orig, 2) == round(ratio_comp, 2):
             #calculate all values, see functions above
             #values are measurements for noise ratio of picture or for similarity between pictures
             #always 2 pictures needed to make comparisons
             mse_value = mse(gray1, gray2)
             ssim_value = compare(gray1, gray2)
             snr = np.mean(10 * np.log10(np.array(image1, dtype=np.float32)**2/(np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2))
             psnr = calculate_psnr(image1,image2)
             correlation = dcor(image1, image2)
             euclid_dist = linalg.norm(image1-image2)
             #print("MSE:", mse_value)
             #print("SSIM:", ssim_value)
             #put all in a long list of items to make calculations on these
             list_item = [c_noise[frame][1], mse_value,ssim_value,snr,psnr, correlation, euclid_dist]
             dirty_data.append(list_item)
             #append 480 values per noise thing but seperated by letter
             for item in letters:
                 for noise in range(len(noise_list)):
                     if c_noise[frame][1] == item and frame < (noise+1)*480 and frame > noise*480:
                         globals()[f"{item}_list_{noise+1}_noiseperc"].append(list_item)               
        
        #for every letter for every noise type, add data to a list
        for noise in range(len(noise_list)):
            if noise == 0:
                globals()[f"dirty{noise+1}_data"] = list(zip(*dirty_data[0:(noise+1)*480]))
            else:
                globals()[f"dirty{noise+1}_data"] = list(zip(*dirty_data[noise*480:(noise+1)*480]))
            for item in letters:
                globals()[f"dirty{noise+1}_data_{item}"] = list(zip(*globals()[f"{item}_list_{noise+1}_noiseperc"]))
        
        #these Qtables are used as overview tables, so that averages of each measure for each noise type/imagery can be presented
        #same averages will be presented in form of box plots later on
        comparenoise_av_noiseperc = QTable(data= None,
                   names=('Noisetype', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with perc stimulus'})
        
        for item in letters:
            globals()[f"comparenoise_{item}_noiseperc"] = QTable(data= None,
                   names=('Noisetype', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with presented stimulus of {item}'})
        globals()[f"comparenoise_imaperc"] = QTable(data= None,
                   names=('Comparison', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with presented stimulus of {item}'})
        globals()[f"comparenoise_averages"] = QTable(data= None,
                   names=('Letter', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with presented stimulus of {item}'})
        globals()[f"comparenoise_ima_vs_avima"] = QTable(data= None,
                   names=('Letter', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Average perception compared with single trial imagery'})
        
        #data for average of all letters for each noise type
        for noise in range(len(noise_list)):
            row_item = [noise_list[noise]]
            for measure in range(len(measures)):
                row_item.append(np.mean(globals()[f"dirty{noise+1}_data"][measure+1]))
            comparenoise_av_noiseperc.add_row(row_item)
        
        #comparenoise_av_noiseperc.add_row(['Noise8000', np.mean(dirty1_data[1]), np.mean(dirty1_data[2]),np.mean(dirty1_data[3]),np.mean(dirty1_data[4]),np.mean(dirty1_data[5]),np.mean(dirty1_data[6])])
        #comparenoise_av_noiseperc.add_row(['noiseR2large', np.mean(dirty2_data[1]), np.mean(dirty2_data[2]),np.mean(dirty2_data[3]),np.mean(dirty2_data[4]),np.mean(dirty2_data[5]),np.mean(dirty2_data[6])])
        #comparenoise_av_noiseperc.add_row(['noiseR1large', np.mean(dirty3_data[1]), np.mean(dirty3_data[2]),np.mean(dirty3_data[3]),np.mean(dirty3_data[4]),np.mean(dirty3_data[5]),np.mean(dirty3_data[6])])
        #comparenoise_av_noiseperc.add_row(['noiseR2', np.mean(dirty4_data[1]), np.mean(dirty4_data[2]),np.mean(dirty4_data[3]),np.mean(dirty4_data[4]),np.mean(dirty4_data[5]),np.mean(dirty4_data[6])])
        #comparenoise_av_noiseperc.add_row(['noiseR1', np.mean(dirty5_data[1]), np.mean(dirty5_data[2]),np.mean(dirty5_data[3]),np.mean(dirty5_data[4]),np.mean(dirty5_data[5]),np.mean(dirty5_data[6])])
        
        print("Noise comparison of perception and perception+noise averaged over all letters")
        #use indicater in printed output to see what is important
        print(matfiles[0][0:3], "\n",comparenoise_av_noiseperc)
        
        #data for average of letters, show table for all every noise type
        for item in letters:
            index = 1
            for noise in noise_list:
                globals()[f"comparenoise_{item}_noiseperc"].add_row([noise,np.mean(globals()[f"dirty{index}_data_{item}"][1]),np.mean(globals()[f"dirty{index}_data_{item}"][2]),np.mean(globals()[f"dirty{index}_data_{item}"][3]),np.mean(globals()[f"dirty{index}_data_{item}"][4]),np.mean(globals()[f"dirty{index}_data_{item}"][5]),np.mean(globals()[f"dirty{index}_data_{item}"][6])])
                index +=1
            print("\n", matfiles[0][0:3], "\n", "Noise comparison of perception and perception+noise of " + item)
            print(globals()[f"comparenoise_{item}_noiseperc"])
        
        #compare perception data to imagery data
        #c = clean, label, dirty
        #c[1] = letter label, so that letter label can be used as name identifier        
        for item in letters:
            for pic in range(len(c_noise)):
                if c_noise[pic][1] == item:
                    globals()[f"{item}_list"].append(c_noise[pic][0])
            for pic in range(len(c_ima)):
                if c_ima[pic][1] == item:
                    globals()[f"{item}_list_ima"].append(c_ima[pic][0])
            #average imagery for each letter, average perception letter
            globals()[f"{item}_list_imamean"] = np.mean(globals()[f"{item}_list_ima"], axis = 0)
            globals()[f"{item}_list_percmean"] = np.mean(globals()[f"{item}_list"],axis = 0)
        
        #average imagery data compared to perception overall and for each letter separately
        #same thing as before just without different noise types but instead clean perception vs average imagery
        for frame in range(len(c_noise)):        
                image1 = c_noise[frame][0] #this is the clean perception frame
                item = c_noise[frame][1]
                image2 = globals()[f"{item}_list_imamean"] # this is the average imagery picture
                gray1 = image1
                gray2= image2
                # Check for same size and ratio and report accordingly
                ho, wo = image1.shape
                hc, wc = image2.shape
                ratio_orig = ho/wo
                ratio_comp = hc/wc
                dim = (wc, hc)
                if round(ratio_orig, 2) != round(ratio_comp, 2):
                 print("\nImages not of the same dimension. Check input.")
                 exit()
                
                # Resize first image if the second image is smaller
                elif ho > hc and wo > wc:
                 print("\nResizing original image for analysis...")
                 gray1 = cv2.resize(gray1, dim)
                
                elif ho < hc and wo < wc:
                 print("\nCompressed image has a larger dimension than the original. Check input.")
                 exit()
                
                if round(ratio_orig, 2) == round(ratio_comp, 2):
                 mse_value = mse(gray1, gray2)
                 ssim_value = compare(gray1, gray2)
                 snr = np.mean(10 * np.log10(np.array(image1, dtype=np.float32)**2/(np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2))
                 psnr = calculate_psnr(image1,image2)
                 correlation = dcor(image1, image2)
                 euclid_dist = linalg.norm(image1-image2)
                 list_item = [c_noise[frame][1], mse_value,ssim_value,snr,psnr, correlation, euclid_dist]
                 dirty_data_ima.append(list_item)
                 globals()[f"dirty_data_{item}_ima"].append(list_item)
        #now do the calculations for each letter independently on every imagery trial vs average imagery
        #check these numbers to see general range of measures         
        for item in letters:
            for frame in range(len(globals()[f"{item}_list_ima"])):        
                image1 = globals()[f"{item}_list_ima"][frame]
                image2 = globals()[f"{item}_list_imamean"]
                gray1 = image1
                gray2= image2
                # Check for same size and ratio and report accordingly
                ho, wo = image1.shape
                hc, wc = image2.shape
                ratio_orig = ho/wo
                ratio_comp = hc/wc
                dim = (wc, hc)
                if round(ratio_orig, 2) != round(ratio_comp, 2):
                 print("\nImages not of the same dimension. Check input.")
                 exit()
                
                # Resize first image if the second image is smaller
                elif ho > hc and wo > wc:
                 print("\nResizing original image for analysis...")
                 gray1 = cv2.resize(gray1, dim)
                
                elif ho < hc and wo < wc:
                 print("\nCompressed image has a larger dimension than the original. Check input.")
                 exit()
                
                if round(ratio_orig, 2) == round(ratio_comp, 2):
                 mse_value = mse(gray1, gray2)
                 ssim_value = compare(gray1, gray2)
                 snr = np.mean(10 * np.log10(np.array(image1, dtype=np.float32)**2/(np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2))
                 psnr = calculate_psnr(image1,image2)
                 correlation = dcor(image1, image2)
                 euclid_dist = linalg.norm(image1-image2)
                 list_item = [c_noise[frame][1], mse_value,ssim_value,snr,psnr, correlation, euclid_dist]
                 dirty_data_ima.append(list_item)
                 globals()[f"data_{item}_ima_vs_av"].append(list_item)
        #av perce vs av imagery for each letter, only one calculation per letter, no average of values
        print("average perception vs average imagery for each letter")
        for item in letters:
            image1 = globals()[f"{item}_list_percmean"]
            image2 = globals()[f"{item}_list_imamean"]
            gray1 = image1
            gray2= image2
            # Check for same size and ratio and report accordingly
            ho, wo = image1.shape
            hc, wc = image2.shape
            ratio_orig = ho/wo
            ratio_comp = hc/wc
            dim = (wc, hc)
            if round(ratio_orig, 2) != round(ratio_comp, 2):
             print("\nImages not of the same dimension. Check input.")
             exit()
            
            # Resize first image if the second image is smaller
            elif ho > hc and wo > wc:
             print("\nResizing original image for analysis...")
             gray1 = cv2.resize(gray1, dim)
            
            elif ho < hc and wo < wc:
             print("\nCompressed image has a larger dimension than the original. Check input.")
             exit()
            
            if round(ratio_orig, 2) == round(ratio_comp, 2):
             mse_value = mse(gray1, gray2)
             ssim_value = compare(gray1, gray2)
             snr = np.mean(10 * np.log10(np.array(image1, dtype=np.float32)**2/(np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32)) ** 2))
             psnr = calculate_psnr(image1,image2)
             correlation = dcor(image1, image2)
             euclid_dist = linalg.norm(image1-image2)
             list_item = [item, mse_value,ssim_value, snr, psnr, correlation, euclid_dist]
             globals()[f"comparenoise_averages"].add_row(list_item)
             #print(list_item)
        
        print("\n", matfiles[0][0:3], "\n",  "Average perception versus average imagery of each letter separately (no mean)")
        print(globals()[f"comparenoise_averages"])                
        dirty_data_ima = list(zip(*dirty_data_ima))
        dirty_data = list(zip(*dirty_data))
          
        comparenoise_av_imaperc = QTable(data= None,
                   names=('Noisetype', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with perc stimulus'})    
            
        #data for average of all letters
        #comparenoise_av_imaperc.add_row(['PercIma', np.mean(dirty_data_ima[1]), np.mean(dirty_data_ima[2]),np.mean(dirty_data_ima[3]),np.mean(dirty_data_ima[4]),np.mean(dirty_data_ima[5]),np.mean(dirty_data_ima[6])])
        
        print("\n", matfiles[0][0:3], "\n","  Comparison of perception (each single trial) with average imagery")
        print(comparenoise_av_imaperc)
        
        #data for average of letters
        globals()[f"comparenoise_imaperc"].add_row(['PercIma', np.mean(dirty_data_ima[1]), np.mean(dirty_data_ima[2]),np.mean(dirty_data_ima[3]),np.mean(dirty_data_ima[4]),np.mean(dirty_data_ima[5]),np.mean(dirty_data_ima[6])])
        for item in letters:
            #add row to the Qtable for nice output, but need to zip before to average out
            globals()[f"dirty_data_{item}_ima"] = list(zip(*globals()[f"dirty_data_{item}_ima"]))
            globals()[f"comparenoise_imaperc"].add_row([item,np.mean(globals()[f"dirty_data_{item}_ima"][1]),np.mean(globals()[f"dirty_data_{item}_ima"][2]),np.mean(globals()[f"dirty_data_{item}_ima"][3]),np.mean(globals()[f"dirty_data_{item}_ima"][4]),np.mean(globals()[f"dirty_data_{item}_ima"][5]),np.mean(globals()[f"dirty_data_{item}_ima"][6])])
            globals()[f"data_{item}_ima_vs_av"] = list(zip(*globals()[f"data_{item}_ima_vs_av"]))
            globals()[f"comparenoise_ima_vs_avima"].add_row([item,np.mean(globals()[f"data_{item}_ima_vs_av"][1]),np.mean(globals()[f"data_{item}_ima_vs_av"][2]),np.mean(globals()[f"data_{item}_ima_vs_av"][3]),np.mean(globals()[f"data_{item}_ima_vs_av"][4]), np.mean(globals()[f"data_{item}_ima_vs_av"][5]), np.mean(globals()[f"data_{item}_ima_vs_av"][6])])
        #indicate name of participant first
        print("\n", matfiles[0][0:3], "\n", globals()[f"comparenoise_imaperc"])
        print("\n", matfiles[0][0:3], "\n", globals()[f"comparenoise_ima_vs_avima"])
        #indicate range of imagery average vs each imagery file, to see maximum range
        for measure in range(len(measures)):
            #print range inside this ae
            print("range of ", measures[measure], " for av imagery vs each ima frame")
            print(np.min(np.array(globals()[f"data_{item}_ima_vs_av"][measure+1]).astype(float)), " - ", np.max(np.array(globals()[f"data_{item}_ima_vs_av"][measure+1]).astype(float)))

        #####stats part
        #compare differences of PercvsNoise with PercvsIma, statistical significance at 0.05
        #do kolmogorov smirnov as well as t-test        
        for noise in range(len(noise_list)):
            print("\n", matfiles[0][0:3], "\n",)
            for measure in range(len(measures)):
                    #all letters at once
                    print("Average of all letters for",noise_list[noise],measures[measure], "vs imagery average")
                    print(stats.ks_2samp(globals()[f"dirty{noise+1}_data"][measure+1],dirty_data_ima[measure+1]))
                    print(stats.ttest_ind(globals()[f"dirty{noise+1}_data"][measure+1],dirty_data_ima[measure+1]))
                    #each letter separately
                    for item in letters:
                        print(item, noise_list[noise], measures[measure],"vs imagery average for", item)
                        print(stats.ks_2samp(globals()[f"dirty{noise+1}_data_{item}"][measure+1],globals()[f"dirty_data_{item}_ima"][measure+1]))
                        print(stats.ttest_ind(globals()[f"dirty{noise+1}_data_{item}"][measure+1],globals()[f"dirty_data_{item}_ima"][measure+1]))
        
        ##### graph distribution of all values
        #bar plots, distributions shown, for each subject separately
        # row: which noise type
        # column: which measure of similarity/noise
        fig_av_noise, axs = pyplot.subplots(len(noise_list), len(measures))
        fig_av_noise.suptitle(f"distributions of similarity measures as averaged over noise data {matfiles[0][0:3]}")
        for noise in range(len(noise_list)):
            for measure in range(len(measures)):
                if len(noise_list)== 1:
                    axs[measure].hist([v for v in globals()[f"dirty{noise+1}_data"][measure+1] if not isinf(v)])
                else:
                    axs[noise, measure].hist([v for v in globals()[f"dirty{noise+1}_data"][measure+1] if not isinf(v)])
                #axs[noise, measure].set_title(noise_list[noise])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
        
        #imagery data plot
        fig_av_ima, axs2 = pyplot.subplots(1, len(measures))
        fig_av_ima.suptitle(f"distributions of similarity measures as averaged over imagery data {matfiles[0][0:3]}")
        for measure in range(len(measures)):
            axs2[measure].hist([v for v in globals()[f"dirty_data_ima"][measure+1] if not isinf(v)])
            axs2[measure].set_title(measures[measure])
            for ax in axs2.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs2.flat:
                ax.label_outer()
        
        #plot distributions for each letter independently
        #each of these plots is accessible with different name, not yet for each participant new files, just run with only one participant to have plots or change names
        #globals() makes variable names out of specific string
        for item in letters:
            globals()[f"fig_{item}_noise"], axs = pyplot.subplots(len(noise_list), len(measures))
            globals()[f"fig_{item}_noise"].suptitle(f"distributions of similarity measure of {item} {matfiles[0][0:3]}")
            for noise in range(len(noise_list)):
                for measure in range(len(measures)):
                    if len(noise_list)== 1:
                        #some inf values were inside those, therefore could not be plotted, now got rid of this
                        axs[measure].hist([v for v in globals()[f"dirty{noise+1}_data_{item}"][measure+1] if not isinf(v)])
                    else:
                        axs[noise, measure].hist([v for v in globals()[f"dirty{noise+1}_data_{item}"][measure+1] if not isinf(v)])
                    #axs[noise, measure].set_title(noise_list[noise])
                for ax in axs.flat:
                    ax.set(xlabel='values', ylabel='number of cases')    
                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()
            globals()[f"fig_{item}_ima"], axs = pyplot.subplots(1, len(measures))
            globals()[f"fig_{item}_ima"].suptitle(f"distributions of similarity measure of {item} for imagery {matfiles[0][0:3]}")
            for measure in range(len(measures)):
                axs[measure].hist([v for v in globals()[f"dirty_data_{item}_ima"][measure+1] if not isinf(v)])
                axs2[measure].set_title(measures[measure])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
                
        #plot average imagery vs imagery each pic
        #bar charts
        for item in letters:
            globals()[f"fig_{item}_ima_vs_av"], axs = pyplot.subplots(1, len(measures))
            globals()[f"fig_{item}_ima_vs_av"].suptitle(f"distributions of similarity measure of {item} for ima vs av ima {matfiles[0][0:3]}")
            for measure in range(len(measures)):
                axs[measure].hist([v for v in globals()[f"data_{item}_ima_vs_av"][measure+1] if not isinf(v)])
                axs2[measure].set_title(measures[measure])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()       
                
        print("++++")
        #print means of each noise type for each measure for every noise type vs imagery 
        #show best noise type, most of the time same type of noise best in all measures
        #best evaluated as smallest difference
        plot_data = []
        matlab_data = []
        #print("Averaged over all letters best noise type per measure \n") 
        print("Average")    
        for measure in range(len(measures)):
            globals()[f"{measures[measure]}_diff"] = []
            for noise in range(len(noise_list)):
                globals()[f"x{noise}"] = [v for v in globals()[f"dirty{noise+1}_data"][measure+1] if not isinf(v)]
                plot_data.append(globals()[f"x{noise}"])
                globals()[f"{measures[measure]}_diff"].append([noise_list[noise], mean([v for v in globals()[f"dirty{noise+1}_data"][measure+1] if not isinf(v)])-mean(globals()[f"dirty_data_ima"][measure+1])])     
            plot_data.append(dirty_data_ima[measure+1])
            globals()[f"fig_means_{measure}"], ax1 = pyplot.subplots()
            ax1.set_title(measures[measure] + " average of all letters " + matfiles[0][0:3])
            ax1.boxplot(plot_data)
            labels = noise_list + ['ima']
            pyplot.xticks(list(range(1,len(noise_list)+2)), labels, size='small')
            plot_data = []
            print(measures[measure], min(globals()[f"{measures[measure]}_diff"]))
            
        #plots for each letter for each measure a boxplot
        #asteriks in name when bonferroni correct significance reached
        plot_data = []
        noise_labels = []
        
        for item in letters:
            #print(f"For {item} best noise type per measure \n")
            print(item)
            for measure in range(len(measures)):
                globals()[f"{measures[measure]}_{item}_diff"] = []
                for noise in range(len(noise_list)):
                    globals()[f"x{noise}"] = [v for v in globals()[f"dirty{noise+1}_data_{item}"][measure+1] if not isinf(v)]
                    #plot data for boxplot needs one line per measure to make multiple boxes that can be pltoted
                    plot_data.append(globals()[f"x{noise}"])
                    globals()[f"{measures[measure]}_{item}_diff"].append([noise_list[noise], mean([v for v in globals()[f"dirty{noise+1}_data_{item}"][measure+1] if not isinf(v)])-mean(globals()[f"dirty_data_{item}_ima"][measure+1])])     
                    #in case the difference between one noise type and imagery reaches significance, the star is added in front of the name on the x axis
                    test = stats.ttest_ind(globals()[f"dirty{noise+1}_data_{item}"][measure+1],globals()[f"dirty_data_{item}_ima"][measure+1])
                
                    if test.pvalue < 0.05/len(noise_list):
                        noise_labels.append("* "+ noise_list[noise]) 
                    else:
                        noise_labels.append(noise_list[noise])
                plot_data.append(globals()[f"dirty_data_{item}_ima"][measure+1])
                globals()[f"fig_means_{measure}_{item}"], ax1 = pyplot.subplots()
                ax1.set_title(measures[measure] +" of " + item + " " + matfiles[0][0:3])
                ax1.boxplot(plot_data)       
                
                labels = noise_labels + ['ima']
                pyplot.xticks(list(range(1,len(noise_labels)+2)), labels, size='small', fontsize=6)
                #print(measures[measure], min(globals()[f"{measures[measure]}_{item}_diff"]))
                plot_data = []
                noise_labels = []
                sys.stdout.write(str(measures[measure])+ str(min(globals()[f"{measures[measure]}_{item}_diff"]))+ "\n")
        
        #in case this is the first subject, then long lists are initialised
        if len(ima_data_long) == 0:
            ima_data_long = dirty_data_ima
            dirty_data_long = dirty_data
            for item in letters:
                globals()[f"ima_data_{item}_long"] = globals()[f"dirty_data_{item}_ima"]
                globals()[f"ima_data_{item}_imavsav_long"] = globals()[f"data_{item}_ima_vs_av"]
            for noise in range(len(noise_list)):
                globals()[f"dirty{noise+1}_data_long"] = globals()[f"dirty{noise+1}_data"]
                for item in letters:
                    globals()[f"dirty{noise+1}_data_{item}_long"] = globals()[f"dirty{noise+1}_data_{item}"]          
        #in case this is subject two or further, then lists are just appended to that
        else:
            ima_data_long = np.append(ima_data_long, dirty_data_ima, 1)
            dirty_data_long = np.append(dirty_data_long, dirty_data, 1)
            globals()[f"ima_data_{item}_long"] = np.append(globals()[f"ima_data_{item}_long"], globals()[f"dirty_data_{item}_ima"],1)
            for noise in range(len(noise_list)):
                globals()[f"dirty{noise+1}_data_long"] = np.append(globals()[f"dirty{noise+1}_data_long"], globals()[f"dirty{noise+1}_data"],1)
                for item in letters:
                    globals()[f"dirty{noise+1}_data_{item}_long"] = np.append(globals()[f"dirty{noise+1}_data_{item}_long"], globals()[f"dirty{noise+1}_data_{item}"],1)
                
    ######## calculations for average of all persons that were analysed
    #only happen in case more than one subect is in the subject list
    if len(sub_list) > 1:
        allsubjects_table = QTable(data= None,
                   names=('Noisetype', 'MSE', 'SSIM', 'SNR', 'PSNR', 'Correlation', 'Euclidean'),
                   dtype=('U1', 'f4', 'f4','f4','f4','f4','f4'),
                   meta={'name': 'Noisecompared with perc stimulus'})
        for noise in range(len(noise_list)):
            y = np.array(globals()[f"dirty{noise+1}_data_long"][1:7])
            #one long table for all subjects with all noise types, each noise type per letter
            allsubjects_table.add_row([noise_list[noise], np.mean(y[0].astype(float)),np.mean(y[1].astype(float)),np.mean(y[2].astype(float)),np.mean(y[3].astype(float)),np.mean(y[4].astype(float)),np.mean(y[5].astype(float))])
            for item in letters:
                z = np.array(globals()[f"dirty{noise+1}_data_{item}_long"][1:7])
                allsubjects_table.add_row([item + " "+ noise_list[noise],np.mean(z[0].astype(float)),np.mean(z[1].astype(float)),np.mean(z[2].astype(float)),np.mean(z[3].astype(float)),np.mean(z[4].astype(float)),np.mean(z[5].astype(float))])
        n = np.array(ima_data_long[1:7])
        allsubjects_table.add_row(["ima av", np.mean(n[0].astype(float)),np.mean(n[1].astype(float)),np.mean(n[2].astype(float)),np.mean(n[3].astype(float)),np.mean(n[4].astype(float)),np.mean(n[5].astype(float))])
        for item in letters:
            q = np.array(globals()[f"ima_data_{item}_long"][1:7])
            allsubjects_table.add_row(["ima "+ item, np.mean(q[0].astype(float)),np.mean(q[1].astype(float)),np.mean(q[2].astype(float)),np.mean(q[3].astype(float)),np.mean(q[4].astype(float)),np.mean(q[5].astype(float))])
        print(allsubjects_table)
    
        ##### graph distribution of all values for all subjects
        # row: which noise type
        # column: which measure of similarity/noise
        #histograms
    
        #after adding all data from all participants, data was converted to string somehow, therefore inside calculations retransferred to float
        fig_av_noise_all, axs = pyplot.subplots(len(noise_list), len(measures))
        fig_av_noise_all.suptitle("distributions of similarity measures as averaged over noise data all subjects")
        for noise in range(len(noise_list)):
            for measure in range(len(measures)):
                if len(noise_list)== 1:
                    axs[measure].hist([v for v in np.array(globals()[f"dirty{noise+1}_data_long"][measure+1]).astype(float) if not isinf(v)])
                else:
                    axs[noise, measure].hist([v for v in np.array(globals()[f"dirty{noise+1}_data_long"][measure+1]).astype(float) if not isinf(v)])
                #axs[noise, measure].set_title(noise_list[noise])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
        
        #imagery data plot for all subjects
        #initialise subplots
        fig_av_ima_all, axs2 = pyplot.subplots(1, len(measures))
        fig_av_ima_all.suptitle(f"distributions of similarity measures as averaged over imagery data all subjects")
        for measure in range(len(measures)):
            axs2[measure].hist([np.array(globals()[f"ima_data_long"][measure+1]).astype(float)])
            axs2[measure].set_title(measures[measure])
            for ax in axs2.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs2.flat:
                ax.label_outer()
        #plot distributions for all subjects per letter        
        for item in letters:
            globals()[f"fig_{item}_noise_all"], axs = pyplot.subplots(len(noise_list), len(measures))
            globals()[f"fig_{item}_noise_all"].suptitle(f"distributions of similarity measure of {item} all subjects")
            for noise in range(len(noise_list)):
                for measure in range(len(measures)):
                    if len(noise_list)== 1:
                        axs[measure].hist([v for v in np.array(globals()[f"dirty{noise+1}_data_{item}_long"][measure+1]).astype(float) if not isinf(v)])
                    else:
                        axs[noise, measure].hist([v for v in np.array(globals()[f"dirty{noise+1}_data_{item}_long"][measure+1]).astype(float) if not isinf(v)])
                    #axs[noise, measure].set_title(noise_list[noise])
                for ax in axs.flat:
                    ax.set(xlabel='values', ylabel='number of cases')    
                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()
            globals()[f"fig_{item}_ima_all"], axs = pyplot.subplots(1, len(measures))
            globals()[f"fig_{item}_ima_all"].suptitle(f"distributions of similarity measure of {item} for imagery all subjects")
            for measure in range(len(measures)):
                axs[measure].hist([v for v in np.array(globals()[f"ima_data_{item}_long"][measure+1]).astype(float) if not isinf(v)])
                axs2[measure].set_title(measures[measure])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
        
        #plot average imagery vs imagery each pic for all subjects
        #bar charts
        for item in letters:
            globals()[f"fig_{item}_ima_vs_av_all"], axs = pyplot.subplots(1, len(measures))
            globals()[f"fig_{item}_ima_vs_av_all"].suptitle(f"distributions of similarity measure of {item} for ima vs av ima all")
            for measure in range(len(measures)):
                axs[measure].hist([v for v in globals()[f"ima_data_{item}_imavsav_long"][measure+1] if not isinf(v)])
                axs2[measure].set_title(measures[measure])
            for ax in axs.flat:
                ax.set(xlabel='values', ylabel='number of cases')    
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
        
        #print means of each noise type for each measure vs imagery for all subjects summarised
        #box plots
        plot_data = []
        matlab_data = []
        #print("Averaged over all letters best noise type per measure \n") 
        print("Average")    
        for measure in range(len(measures)):
            globals()[f"{measures[measure]}_diff_all"] = []
            for noise in range(len(noise_list)):
                globals()[f"x{noise}_all"] = [np.array(globals()[f"dirty{noise+1}_data_long"][measure+1]).astype(float)]
                plot_data.append(globals()[f"x{noise}"])
                globals()[f"{measures[measure]}_diff_all"].append([noise_list[noise], mean([np.array(globals()[f"dirty{noise+1}_data_long"][measure+1]).astype(float)])-mean(np.array(ima_data_long[measure+1]).astype(float))])     
            plot_data.append(dirty_data_ima[measure+1])
            globals()[f"fig_means_{measure}_all"], ax1 = pyplot.subplots()
            ax1.set_title(measures[measure] + ": average of all letters all subjects")
            ax1.boxplot(plot_data)
            labels = noise_list + ['ima']
            pyplot.xticks(list(range(1,len(noise_list)+2)), labels, size='small')
            plot_data = []
            print(measures[measure], min(globals()[f"{measures[measure]}_diff_all"]))
        
        #plots for each letter for each measure a boxplot for all subjects averaged
        #asteriks in name(below specific box) when bonferroni corrected significance reached
        plot_data = []
        noise_labels = []        
        for item in letters:
            #print(f"For {item} best noise type per measure \n")
            print(item)
            for measure in range(len(measures)):
                globals()[f"{measures[measure]}_{item}_diff_all"] = []
                for noise in range(len(noise_list)):
                    globals()[f"x{noise}_all"] = globals()[f"dirty{noise+1}_data_{item}_long"][measure+1].astype(float)
                    plot_data.append(globals()[f"x{noise}_all"])
                    globals()[f"{measures[measure]}_{item}_diff_all"].append([noise_list[noise], mean([globals()[f"dirty{noise+1}_data_{item}_long"][measure+1].astype(float)])-mean(np.array(globals()[f"ima_data_{item}_long"][measure+1]).astype(float))])     
            
                    test = stats.ttest_ind(globals()[f"dirty{noise+1}_data_{item}_long"][measure+1].astype(float),np.array(globals()[f"ima_data_{item}_long"][measure+1]).astype(float))
                
                    if test.pvalue < 0.05/len(noise_list):
                        noise_labels.append("* "+ noise_list[noise]) 
                    else:
                        noise_labels.append(noise_list[noise])
                plot_data.append(np.array(globals()[f"ima_data_{item}_long"][measure+1]).astype(float))
                globals()[f"fig_means_{measure}_{item}_all"], ax1 = pyplot.subplots()
                ax1.set_title(measures[measure] +" of all subjects of " + item)
                ax1.boxplot(plot_data)       
                
                labels = noise_labels + ['ima']
                pyplot.xticks(list(range(1,len(noise_labels)+2)), labels, size='small', fontsize=6)
                #print(measures[measure], min(globals()[f"{measures[measure]}_{item}_diff"]))
                plot_data = []
                noise_labels = []
                #calculate least bad noise type plus difference that is still there
                #this difference could be used to improve noise, try to decrease difference as much as possible
                sys.stdout.write(str(measures[measure])+ str(min(globals()[f"{measures[measure]}_{item}_diff_all"]))+ "\n")
