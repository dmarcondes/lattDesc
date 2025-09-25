#Process data for lattice descent algorithms
import pandas as pd
from itertools import product
from contextlib import redirect_stdout
import plotly.express as px
import plotly
from lattDesc import utils as ut
import math
import os
from PIL import Image
from IPython.display import display
import numpy as np

#Generate binary synthetic data
def synthetic_binary_data(n,d,key):
    """
    Generate synthetic binary random data. The probability of each point and the respective conditional probability of the labels are uniformly selected.
    -------
    Parameters
    ----------
    n : int

        Sample size

    d : int

        Dimension of the input

    key : int

        Seed for sampling

    Returns
    -------
    numpy.array with the generated data in which the last column constains the labels
    """
    #Generate probability of each point
    rng = np.random.default_rng(seed = key)
    prob = rng.uniform(size = (2 ** d,),low = 0,high = 100)
    prob = prob/np.sum(prob)
    #Generate probability of conditioned distribution
    prob_cond = rng.choice(np.array([0,1]),size = (2 ** d,))
    #Generate input points
    index = rng.choice(np.array(list(range(2 ** d))), size = (n,),replace = True,p = prob)
    domain = np.array([i for i in product(range(2),repeat = d)])
    x = domain[index,:]
    #Generate output points
    y = []
    for i in range(n):
        p = np.array([1 - prob_cond[index[i]],prob_cond[index[i]]])
        y = y + [rng.choice(np.arange(2),size = (1,),p = p)]
    y = np.array(y)
    return np.append(x,y,1)

#Get frequency of a point in the domain
def get_fpoint(x,data,num_classes = 2):
    """
    Get empirical frequencies conditioned on a point
    -------
    Parameters
    ----------
    x : numpy.array

        Point to condition on

    data : numpy.array

        Data array in which the last column contains the labels

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array with the empirical frequencies
    """
    data_cond = data[(data[:,:-1] == x).all(1),:]
    freq = np.bincount(data_cond[:,-1],minlength = num_classes)
    return freq

#Get frequency table
def get_ftable(data,unique,num_classes = 2):
    """
    Get empirical frequencies of data
    -------
    Parameters
    ----------
    data : numpy.array

        Data array in which the last column contains the labels

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array with the frequency table
    """
    #Unique
    if unique:
        domain = data[:,:-1]
        freq = np.where(data[:,-1] == 0,1,0).reshape((domain.shape[0],1))
        for c in range(num_classes-1):
            freq = np.append(freq,np.where(data[:,-1] == c+1,1,0).reshape((domain.shape[0],1)),1)
    #Not unique
    else:
        domain = np.unique(data[:,:-1],axis = 0)
        freq = np.apply_along_axis(lambda x: get_fpoint(x,data,num_classes),1,domain)
    return np.append(domain,np.array(freq).reshape((domain.shape[0],num_classes)),1)

#Generate picture of partition
def picture_partition(intervals,block,title = 'abc',filename = 'image'):
    """
    Generate image of Boolean lattice colored by a given partition partition and save in a .pdf file
    -------
    Parameters
    ----------
    intervals : numpy.array

        Intervals of partition

    block : numpy.array

        Block of each interval

    title : str

        Title of image

    filename : str

        Name of image without extension
    """
    #Generate lattice
    d = intervals.shape[1]
    lattice = np.array([i for i in product(range(2),repeat = d)])
    lattice = lattice[np.sum(lattice,1).argsort()]
    #Get block of each point
    block = np.sum(np.apply_along_axis(lambda point: np.where(ut.get_interval(point,intervals),block,0),1,lattice),1)
    jump = 0.75*math.comb(d,round(d/2))/d
    #Create .tex file
    with open(filename + '.tex', 'w') as f:
        with redirect_stdout(f):
            print('\\documentclass[crop,tikz]{standalone}\n')
            print('\\begin{document}')
            print('\\begin{tikzpicture}[scale=1, transform shape]\n')
            #Create node type for each block
            ncolors = np.max(block) + 1
            if ncolors != 1:
                colors = px.colors.n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', ncolors, colortype = 'rgb')
            else:
                colors = ['rgb(0, 0, 255)']
            colors = plotly.colors.convert_colors_to_same_type(colors,colortype = 'tuple')[0]
            for i in range(np.max(block) + 1):
                red = int(round(colors[i][0]*255))
                green = int(round(colors[i][1]*255))
                blue = int(round(colors[i][2]*255))
                print('\\tikzstyle{b' + str(i) + '} = [rectangle,opacity = .5,draw=black, rounded corners,fill = {rgb:red,' + str(red) + ';green,' + str(green) + ';blue,' + str(blue) + '}]')
            #Title
            print('\\node[above] at (0,0.5) {' + title + '};')
            #Print points
            print('\n')
            vars = 0
            counter = 0
            save_id = []
            for i in range(lattice.shape[0]):
                st = '\\node[b' + str(block[i]) + '] at ('
                tmp_vars = np.sum(lattice[i])
                if tmp_vars != vars:
                    vars = tmp_vars
                    counter = -math.comb(d,vars) + 1
                id = str(lattice[i,:])
                id = id.replace('[','')
                id = id.replace(']','')
                id = id.replace(' ','')
                st = st + str(counter) + ',' + str(-jump*vars) + ') (n' + id + ') {\\tiny ' + id + '};'
                counter = counter + 2
                print(st)
            #Lines
            print('\n')
            print('\\begin{scope}')
            for i in range(lattice.shape[0]):
                id1 = str(lattice[i,:])
                id1 = id1.replace('[','')
                id1 = id1.replace(']','')
                id1 = id1.replace(' ','')
                interval_i = np.where(ut.get_interval(lattice[i,:],intervals))[0]
                for j in range(d):
                    if lattice[i,j] == 0:
                        tmp = lattice.copy()
                        tmp[i,j] = 1 - tmp[i,j]
                        id2 = str(tmp[i,:])
                        id2 = id2.replace('[','')
                        id2 = id2.replace(']','')
                        id2 = id2.replace(' ','')
                        interval_tmp = np.where(ut.get_interval(tmp[i,:],intervals))[0]
                        if interval_tmp == interval_i:
                            red = int(round(colors[block[i]][0]*255))
                            green = int(round(colors[block[i]][1]*255))
                            blue = int(round(colors[block[i]][2]*255))
                            print('\\draw[-,color = {rgb:red,' + str(red) + ';green,' + str(green) + ';blue,' + str(blue) + '}] (n' + id1 + ') -- (n' + id2 + ');')
                        else:
                            print('\\draw[-,opacity=0.2] (n' + id1 + ') -- (n' + id2 + ');')
            print('\\end{scope}')
            print('\\end{tikzpicture}')
            print('\\end{document}')
    os.system('pdflatex ' + filename + '.tex > /dev/null')
    os.system('rm *.tex *.log *.aux > /dev/null')

#Read and organize a data file
def read_data_frame(file,sep = None,header = 'infer',sheet = 0):
    """
    Read a data file and convert to numpy array
    -------
    Parameters
    ----------
    file : str

        File name with extension .csv, .txt, .xls or .xlsx

    sep : str

        Separation character for .csv and .txt files. Default ',' for .csv and ' ' for .txt

    header : int, Sequence of int, ‘infer’ or None

        See pandas.read_csv documentation. Default 'infer'

    sheet : int

        Sheet number for .xls and .xlsx files. Default 0

    Returns
    -------
    numpy.array
    """

    #Find out data extension
    ext = file.split('.')[1]

    #Read data frame
    if ext == 'csv':
        if sep is None:
            sep = ','
        dat = pandas.read_csv(file,sep = sep,header = header)
    elif ext == 'txt':
        if sep is None:
            sep = ' '
        dat = pandas.read_table(file,sep = sep,header = header)
    elif ext == 'xls' or ext == 'xlsx':
        dat = pandas.read_excel(file,header = header,sheet_name = sheet)

    #Convert to mnumpy array structure
    dat = np.array(dat)

    return dat

#Read images into an array
def image_to_np(files_path,binary = False):
    """
    Read an image file and convert to numpy array
    -------
    Parameters
    ----------
    files_path : list of str

        List with the paths of the images to read

    binary : logical

        Whether the image should be binarized

    Returns
    -------
    numpy.array
    """
    dat = None
    for f in files_path:
        img = Image.open(f)
        img = np.array(img)/255
        if len(img.shape) == 3:
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        else:
            img = img.reshape((1,img.shape[0],img.shape[1]))
        if binary:
            img[img <= 0.5] = 0
            img[img > 0.5] = 1
        if dat is None:
            dat = img
        else:
            dat = np.append(dat,img,0)
    return dat

#Save images
def save_images(images,files_path):
    """
    Save images in a nunmpy array to file
    -------
    Parameters
    ----------
    images : numpy.array

        Array of images

    files_path : list of str

        List with the paths of the images to write
    """
    if len(files_path) > 1:
        for i in range(len(files_path)):
            if len(images.shape) == 4:
                tmp = Image.fromarray(np.uint8(np.round(255*images[i,:,:,:]))).convert('RGB')
            else:
                tmp = Image.fromarray(np.uint8(np.round(255*images[i,:,:])))
            tmp.save(files_path[i])
    else:
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(np.round(255*images[0,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(np.round(255*images[0,:,:])))
        tmp.save(files_path[0])

#Print images
def print_images(images):
    """
    Print images
    -------
    Parameters
    ----------
    images : numpy.array

        Array of images
    """
    for i in range(images.shape[0]):
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(np.round(255*images[i,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(np.round(255*images[i,:,:]))).convert('RGB')
        display(tmp)

#Create an index array for an array
def index_array(shape):
    """
    Create a 2D array with the indexes of an array with given 2D shape
    -------
    Parameters
    ----------
    shape : list

        List with shape of 2D array

    Returns
    -------
    numpy.array
    """
    return np.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Process binary images by window W at a coordinate (input should be padded)
def process_window_coord(coord,input,output,W,pad):
    """
    Process input and output binary images by a window W at a coordinate
    -------
    Parametersfixed
    ----------
    coord : numpy.array

        A coordinate array

    input : numpy.array

        Input image. It should be padded with zeroes

    output : numpy.array

        Output image

    W : numpy.array

        An array with the coordinates of the window

    pad : int

        Padding parameter

    Returns
    -------
    numpy.array
    """
    i = coord[0]
    j = coord[1]
    vars = np.array([])
    for k in range(W.shape[0]):
        vars = np.append(vars,input[pad + i + W[k,1],pad + j + W[k,0]])
    vars = np.append(vars,output[i,j])
    return vars

#Process binary images by window W (input is already padded)
def process_window(input,output,index,W,pad):
    """
    Process input and output binary images by a window W
    -------
    Parameters
    ----------
    input : numpy.array

        Input image. It should be padded with zeroes

    output : numpy.array

        Output image

    index : numpy.array

        Array with the indexes of the images

    W : numpy.array

        An array with the coordinates of the window

    pad : int

        Padding parameter

    Returns
    -------
    numpy.array
    """
    return np.apply_along_axis(lambda coord: process_window_coord(coord,input,output,W,pad),1,index)

#Process batch of binary images by window W
def process_window_batch(input,output,W,pad):
    """
    Process a batch of input and output binary images by a window W at a coordinate
    -------
    Parameters
    ----------
    coord : numpy.array

        A coordinate array

    input : numpy.array

        Input images. It should be padded with zeroes

    output : numpy.array

        Output images

    W : numpy.array

        An array with the coordinates of the window

    pad : int

        Padding parameter

    Returns
    -------
    numpy.array
    """
    padded = np.pad(input,((0,0),(pad,pad),(pad,pad)), constant_values = 0)
    index = index_array((input.shape[1],input.shape[2]))
    data = process_window(padded[0,:,:],output[0,:,:],index,W,pad)
    for i in range(output.shape[0] - 1):
        data = np.append(data,process_window(padded[i,:,:],output[i,:,:],index,W,pad),0)
    return data.astype('int32')
