#Process data for lattice descent algorithms
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
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
    Generate synthetic binary random data
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
    jax.numpy.array withe genrated data in which the last column constains the labels
    """
    #Generate probability of each point
    key = jax.random.split(jax.random.PRNGKey(key),n+3)
    prob = jax.random.uniform(key = jax.random.PRNGKey(key[0,0]),shape = (2 ** d,),minval = 0,maxval = 100)
    prob = prob/jnp.sum(prob)
    #Generate probability of conditioned distribution
    prob_cond = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array([0,1]),shape = (2 ** d,))
    #Generate input points
    index = jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array(list(range(2 ** d))), shape=(n,),replace = True,p = prob)
    domain = jnp.array([i for i in product(range(2),repeat = d)])
    x = domain[index,:]
    #Generate output points
    y = []
    for i in range(n):
        p = jnp.array([1 - prob_cond[index[i].tolist()],prob_cond[index[i].tolist()]])
        y = y + [jax.random.choice(jax.random.PRNGKey(key[i + 3,0]), jnp.array(list(range(2))),shape=(1,),p = p).tolist()]
    y = jnp.array(y)
    return jnp.append(x,y,1)

#Get frequency of a point in the domain
def get_fpoint(x,data,num_classes = 2):
    """
    Get empirical frequencies conditioned on a point
    -------
    Parameters
    ----------
    x : jax.numpy.array

        Point to condition on

    data : jax.numpy.array

        Data array

    num_classes : int

        Number of classes

    Returns
    -------
    jax.numpy.array with the empirical frequencies
    """
    data = jax.lax.select(jnp.repeat((data[:,:-1] == x).all(1).reshape((data.shape[0],1)),data.shape[1],1),data,-1 + jnp.zeros(data.shape).astype(data.dtype))
    freq = jax.nn.one_hot(data[:,-1],num_classes)
    return jnp.sum(freq,0)

get_fpoint = jax.jit(get_fpoint,static_argnames = ['num_classes'])

#Get frequence table
def get_ftable(data,unique,num_classes = 2):
    """
    Get empirical frequencies of data with binary outputs
    -------
    Parameters
    ----------
    data : jax.numpy.array

        Data array

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    num_classes : int

        Number of classes

    Returns
    -------
    jax.numpy.array
    """
    #Get domain
    if not unique:
        domain = jnp.unique(data[:,:-1],axis = 0)
    else:
        domain = data[:,:-1]
    #Compute frequencies
    f = jax.vmap(lambda x: get_fpoint(x,data,num_classes))(domain)
    return np.array(jnp.append(domain,f,1).astype(data.dtype))

#Generate picture of paritiion
def picture_partition(intervals,block,title = 'abc',filename = 'image'):
    """
    Generate image of Boolean lattice colored by the partition
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Intervals of partition

    block : jax.numpy.array

        Block of each interval

    title : str

        Title of image

    filename : str

        Name of image withou extension
    """
    #Generate lattice
    d = intervals.shape[1]
    lattice = jnp.array([i for i in product(range(2),repeat = d)])
    lattice = lattice[jnp.sum(lattice,1).argsort()]
    #Get block of each point
    block = jnp.sum(jax.vmap(lambda point: jnp.where(ut.get_interval(point,intervals),block,0))(lattice),1)
    jump = 0.75*math.comb(d,round(d/2))/d
    #Create .tex file
    with open(filename + '.tex', 'w') as f:
        with redirect_stdout(f):
            print('\\documentclass[crop,tikz]{standalone}\n')
            print('\\begin{document}')
            print('\\begin{tikzpicture}[scale=1, transform shape]\n')
            #Create node type for each block
            ncolors = jnp.max(block) + 1
            if ncolors != 1:
                colors = px.colors.n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', ncolors, colortype = 'rgb')
            else:
                colors = ['rgb(0, 0, 255)']
            colors = plotly.colors.convert_colors_to_same_type(colors,colortype = 'tuple')[0]
            for i in range(jnp.max(block) + 1):
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
                tmp_vars = jnp.sum(lattice[i])
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
                interval_i = jnp.where(ut.get_interval(lattice[i,:],intervals))[0]
                for j in range(d):
                    if lattice[i,j] == 0:
                        tmp = lattice.at[i,j].set(1 - lattice[i,j])
                        id2 = str(tmp[i,:])
                        id2 = id2.replace('[','')
                        id2 = id2.replace(']','')
                        id2 = id2.replace(' ','')
                        interval_tmp = jnp.where(ut.get_interval(tmp[i,:],intervals))[0]
                        if interval_tmp == interval_i:
                            red = int(round(colors[block[i]][0]*255))
                            green = int(round(colors[block[i]][1]*255))
                            blue = int(round(colors[block[i]][2]*255))
                            print('\\draw[-,color = {rgb:red,' + str(red) + ';green,' + str(green) + ';blue,' + str(blue) + '}] (n' + id1 + ') -- (n' + id2 + ');')
                        else:
                            print('\\draw[-] (n' + id1 + ') -- (n' + id2 + ');')
            print('\\end{scope}')
            print('\\end{tikzpicture}')
            print('\\end{document}')
    os.system('pdflatex ' + filename + '.tex > /dev/null')
    os.system('rm *.tex *.log *.aux > /dev/null')

#Read and organize a data.frame
def read_data_frame(file,sep = None,header = 'infer',sheet = 0):
    """
    Read a data file and convert to JAX array
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
    jax.numpy.array
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

    #Convert to JAX data structure
    dat = jnp.array(dat,dtype = jnp.float32)

    return dat

#Read images into an array
def image_to_jnp(files_path,binary = False):
    """
    Read an image file and convert to JAX array
    -------
    Parameters
    ----------
    files_path : list of str

        List with the paths of the images to read

    binary : logical

        Whether the image should be binarized

    Returns
    -------
    jax.numpy.array
    """
    dat = None
    for f in files_path:
        img = Image.open(f)
        img = jnp.array(img,dtype = jnp.float32)/255
        if len(img.shape) == 3:
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        else:
            img = img.reshape((1,img.shape[0],img.shape[1]))
        if binary:
            img = img.at[img <= 0.5].set(0)
            img = img.at[img > 0.5].set(1)
        if dat is None:
            dat = img
        else:
            dat = jnp.append(dat,img,0)
    return dat

#Save images
def save_images(images,files_path):
    """
    Save images in a JAX array to file
    -------
    Parameters
    ----------
    images : jax.numpy.array

        Array of images

    files_path : list of str

        List with the paths of the images to write
    """
    if len(files_path) > 1:
        for i in range(len(files_path)):
            if len(images.shape) == 4:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
            else:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:])))
            tmp.save(files_path[i])
    else:
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:])))
        tmp.save(files_path[0])

#Print images
def print_images(images):
    """
    Print images
    -------
    Parameters
    ----------
    images : jax.numpy.array

    Array of images
    """
    for i in range(images.shape[0]):
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:]))).convert('RGB')
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
    jax.numpy.array
    """
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Process binary images by window W at a coordinate (input should be padded)
@jax.jit
def process_window_coord(coord,input,output,W,pad):
    """
    Process a binary image by a window W at a coordinate
    -------
    Parameters
    ----------
    coord : jax.numpy.array

        A coordinate array

    input : jax.numpy.array

        Input image. It shoould be padded with zeroes

    output : jax.numpy.array

        Output image

    W : jax.numpy.array

        An array with the coordinates of the window

    pad : int

        List with the paths of the images to write

    Returns
    -------
    jax.numpy.array
    """
    i = coord[0]
    j = coord[1]
    vars = jnp.array([])
    for k in range(W.shape[0]):
        vars = jnp.append(vars,input[pad + i + W[k,1],pad + j + W[k,0]])
    vars = jnp.append(vars,output[i,j])
    return vars

#Process binary images by window W (input is already padded)
@jax.jit
def process_window(input,output,index,W,pad):
    """
    Process a binary image by a window W
    -------
    Parameters
    ----------
    input : jax.numpy.array

        Input image. It shoould be padded with zeroes

    output : jax.numpy.array

        Output image

    W : jax.numpy.array

        An array with the coordinates of the window

    pad : int

        List with the paths of the images to write

    Returns
    -------
    jax.numpy.array
    """
    return jax.vmap(lambda coord: process_window_coord(coord,input,output,W,pad))(index)

#Process batch of binary images by window W
def process_window_batch(input,output,W,pad):
    """
    Process a batch of binary images by a window W at a coordinate
    -------
    Parameters
    ----------
    coord : jax.numpy.array

        A coordinate array

    input : jax.numpy.array

        Input image. It shoould be padded with zeroes

    output : jax.numpy.array

        Output image

    W : jax.numpy.array

        An array with the coordinates of the window

    pad : int

        List with the paths of the images to write

    Returns
    -------
    jax.numpy.array
    """
    padded = jax.lax.pad(input,0.0,((0,0,0),(pad,pad,0),(pad,pad,0)))
    index = index_array((input.shape[1],input.shape[2]))
    data = jax.vmap(lambda input,output: process_window(input,output,index,W,pad))(padded,output)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    return data
