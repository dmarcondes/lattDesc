#Utility functions lattDesc
import numpy as np
import time
from lattDesc import data as dt
import math

import jax #Erase

#Get frequency table of batch
def get_tfrequency_batch(b,batches,tab_train,tab_val,train,val,bsize,bsize_val,unique,batch_val,nval,num_classes = 2):
    """
    Get frequency tables of a batch
    -------

    Parameters
    ----------
    b : int

        Index of batch

    batches : int

        Number of batches

    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    train : numpy.array

        Array with training data. The last column contains the labels

    val : numpy.array

        Array with validation data. The last column contains the labels

    bsize : int

        Batch size of training

    bsize_val : int

        Batch size of validation

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    batch_val : logical

        Whether to consider batches for the validation data

    nval : int

        Size of validation sample

    num_classes : int

        Number of classes

    Returns
    -------
        numpy.arrays of tab_train_batch and tab_val_batch and the batch validation sample size bnval
    """
    if batches > 1:
        if b < batches - 1: #If it is not last batch
            if unique: #If data is unique, batch of frequency table
                tab_train_batch = tab_train[(b*bsize):((b+1)*bsize),:] #Get frequency table of batch
            else: #Else, compute frequency table of data batch
                tab_train_batch = dt.get_ftable(train[(b*bsize):((b+1)*bsize),:],unique,num_classes) #Compute frequency table of batch
        else: #For the last batch
            if unique: #If data is unique, batch of frequency table
                tab_train_batch = tab_train[(b*bsize):,:] #Get frequency table of batch
            else: #Else, compute frequency table of data batch
                tab_train_batch = dt.get_ftable(train[(b*bsize):,:],unique,num_classes) #Compute frequency table of batch
        if batch_val: #If batches for validation should be considered
            if b < batches - 1: #If it is not last batch
                if unique: #If data is unique, batch of frequency table
                    tab_val_batch = tab_val[(b*bsize_val):((b+1)*bsize_val),:] #Get frequency table of batch
                else: #Else, compute frequency table of data batch
                    tab_val_batch = dt.get_ftable(val[(b*bsize_val):((b+1)*bsize_val),:],unique,num_classes) #Compute frequency table of batch
            else: #For the last batch
                if unique: #If data is unique, batch of frequency table
                    tab_val_batch = tab_val[(b*bsize_val):,:] #Get frequency table of batch
                else: #Else, compute frequency table of data batch
                    tab_val_batch = dt.get_ftable(val[(b*bsize_val):,:],unique,num_classes) #Compute frequency table of batch
            bnval = np.sum(tab_val_batch[:,-num_classes:])
        else: #No batch for validation
            tab_val_batch = tab_val #Copy frequency table
            bnval = nval #Copy validation sample size
        return tab_train_batch,tab_val_batch,bnval
    else:
        return tab_train,tab_val,nval

#Teste partial order
def partial_order(x,y):
    """
    Test if x <= y
    -------
    Parameters
    ----------
    x : numpy.array

        Point

    y : numpy.array

        Point

    Returns
    -------
    logical
    """
    return np.sum(x <= y) == x.shape[0]

#Test if element is in interval
def test_interval(interval,x):
    """
    Test if element is in interval
    -------
    Parameters
    ----------
    interval : numpy.array

        Interval

    x : numpy.array

        Element

    Returns
    -------
    logical
    """
    return np.sum(x[interval != -1] != interval[interval != -1]) == 0

#Test if element is not in interval
def test_not_interval(interval,x):
    """
    Test if element is not in interval
    -------
    Parameters
    ----------
    interval : numpy.array

        Interval

    x : numpy.array

        Element

    Returns
    -------
    logical
    """
    return not test_interval(interval,x)

#Test if element is limit of interval (we know it is in interval)
def test_limit_interval(interval,x):
    """
    Test if element is the limit of interval
    -------
    Parameters
    ----------
    interval : numpy.array

        Interval

    x : numpy.array

        Element

    Returns
    -------
    logical
    """
    x_max = np.where(interval < 0,1,interval)
    x_min = np.where(interval < 0,0,interval)
    return (np.sum(x == x_max) == x.shape[0]) or (np.sum(x == x_min) == x.shape[0])

#Get limits of interval
def get_limits_interval(interval,data):
    """
    Flag the points in the data that are limits of interval
    -------
    Parameters
    ----------
    interval : numpy.array

        Interval

    data : numpy.array

        Data

    Returns
    -------
    numpy.array of logical
    """
    return np.apply_along_axis(lambda x: test_limit_interval(interval,x),1,data)

#Get limits of each interval
def get_limits_each_interval(intervals,data):
    """
    Flag the elements in dataset that are the limits of each interval
    -------

    Parameters
    ----------
    intervals : numpy.array

        Intervals

    data : numpy.array

        Dataset

    Returns
    -------

    jax.numpy.array of logical
    """
    return np.apply_along_axis(lambda interval: get_limits_interval(interval,data),1,intervals)

#Get elements that are limit of some interval
def get_limits_some_interval(intervals,data):
    """
    Flag the points in the data that are limits of some interval in a array
    -------
    Parameters
    ----------
    intervals : numpy.array

        Array of intervals

    data : numpy.array

        Data

    Returns
    -------
    numpy.array of logical
    """
    return np.sum(get_limits_each_interval(intervals,data),0) > 0

#Flag interval that contain point
def get_interval(point,intervals):
    """
    Flag interval that constains a point
    -------
    Parameters
    ----------
    point : numpy.array

        Point

    intervals : numpy.array

        Intervals

    Returns
    -------
    numpy.array of logical
    """
    return np.apply_along_axis(lambda interval: test_interval(interval,point),1,intervals)

#Get elements in interval
def get_elements_interval(interval,data):
    """
    Flag the elements in dataset that are in an interval
    -------

    Parameters
    ----------
    interval : numpy.array

        Interval

    data : numpy.array

        Dataset

    Returns
    -------
    numpy.array of logical
    """
    return np.apply_along_axis(lambda x: test_interval(interval,x),1,data)

#Get elements in each interval
def get_elements_each_interval(intervals,data):
    """
    Flag the elements in dataset that are in each interval
    -------

    Parameters
    ----------
    intervals : numpy.array

        Intervals

    data : numpy.array

        Dataset

    Returns
    -------
    numpy.array of logical
    """
    return np.apply_along_axis(lambda interval: get_elements_interval(interval,data),1,intervals)

#Get elements in some interval
def get_elements_some_interval(intervals,data):
    """
    Flag the elements in dataset that are in some interval in an array
    -------
    Parameters
    ----------
    intervals : numpy.array

        Array of intervals

    data : numpy.array

        Data

    Returns
    -------
    numpy.array of logical
    """
    return np.sum(get_elements_each_interval(intervals,data),0) > 0

#Compute error
def val_error(tab_train,tab_val,nval,rng,num_classes = 2):
    """
    Compute the validation error
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    nval : int

        Size of validation sample

    rng : numpy.random

        Random generator object

    num_classes : int

        Number of classes

    Returns
    -------
    float
    """
    freq = np.sum(tab_train[:,-num_classes:],0)
    pred = freq == np.max(freq)
    p = np.where(pred,1,0)
    pred = rng.choice(np.arange(num_classes),size = (1,),p = p/np.sum(p))
    freq_val = np.sum(tab_val[:,-num_classes:],0)
    err = np.sum(freq_val[np.arange(num_classes) != pred])
    return err/nval

#Frequency table of block
def ftable_block(intervals,tab,num_classes = 2):
    """
    Frequency table of a block
    -------
    Parameters
    ----------
    intervals : numpy.array

        Array of intervals that define the block

    tab : numpy.array

        Frequency table. Each row refers to an input point and the num_classes columns refer to label frequencies

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array
    """
    tab_block = tab[get_elements_some_interval(intervals,tab[:,0:-num_classes,]),:]
    return np.sum(tab_block[:,-num_classes:],0)

#Get label of each interval
def estimate_label_block(tab_train,intervals,rng,num_classes = 2):
    """
    Estimate the label of an interval from training data
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : numpy.array

        Intervals of block

    rng : numpy.random

        Random generator object

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array
    """
    tab_train = ftable_block(intervals,tab_train,num_classes)
    pred = freq == np.max(freq)
    p = np.where(pred,1,0)
    pred = rng.choice(np.arange(num_classes),size = (1,),p = p/np.sum(p))
    return pred

#Estimate the label of a partition
def estimate_label_partition(tab_train,intervals,block,rng,num_classes = 2):
    """
    Estimate the label of a partition from training data
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : numpy.array

        Array of intervals that define the partition

    block : numpy.array

        Array with the block of each interval

    rng : numpy.random

        Random generator object

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array
    """
    pred = []
    for i in range(np.max(block) + 1):
        tmp_intervals = intervals[block == i,:]
        pred_block = estimate_label_block(tab_train,tmp_intervals,rng,num_classes)
        pred = pred + [pred_block[0]]
    return np.array(pred)

#Get estimated function
def get_estimated_function(tab_train,intervals,block,rng,num_classes = 2):
    """
    Get estimated function of partition
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : numpy.array

        Array of intervals that define the partition

    block : numpy.array

        Array with the block of each interval

    rng : numpy.random

        Random generator object

    num_classes : int

        Number of classes

    key : int

        Seed for random classification in the presence of ties

    Returns
    -------
    jax.numpy.array
    """
    label_blocks = estimate_label_partition(tab_train,intervals,block,rng,num_classes)
    def f(point):
        return label_blocks[get_interval(point,intervals)]
    return f

#Get error partition
def error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes = 2):
    """
    Get error of partition
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : numpy.array

        Array of intervals that define the partition

    block : numpy.array

        Array with the block of each interval

    nval : int

        Size of validation sample

    rng : numpy.random

        Random generator object

    num_classes : int

        Number of classes

    Returns
    -------
    float
    """
    error = 0
    for i in range(np.max(block) + 1):
        tmp_intervals = intervals[block == i,:]
        tab_train_block = ftable_block(tmp_intervals,tab_train,num_classes)
        tab_val_block = ftable_block(tmp_intervals,tab_val,num_classes)
        error = error + val_error(tab_train_block.reshape((1,num_classes)),tab_val_block.reshape((1,num_classes)),nval,rng,num_classes)
    return error

#Break interval at new interval
def cover_break_interval(new_interval,where_fill):
    """
    Compute a cover of [A,B]/[A,X] or [A,B]/[X,B] by intervals
    -------
    Parameters
    ----------
    new_interval : numpy.array

        The interval [A,X] or [X,B]

    where_fill : numpy.array

        Pre-computed random ordering of the free positions of [A,B] that are not free in new_interval

    Returns
    -------
    numpy.array of intervals
    """
    cover_intervals = None
    for i in range(len(where_fill)):
        tmp = new_interval.copy()
        tmp[0,where_fill[i]] = 1 - tmp[0,where_fill[i]]
        tmp[0,where_fill[(i+1):]] = -1
        if cover_intervals is None:
            cover_intervals = tmp
        else:
            cover_intervals = np.append(cover_intervals,tmp,0)
    cover_intervals = np.append(cover_intervals,new_interval,0)
    return cover_intervals

#Get interval as sup
def get_sup(point,interval):
    """
    Get interval [A,X] given interval [A,B] and point X in [A,B]
    -------
    Parameters
    ----------
    point : numpy.array

        Point

    interval : numpy.array

        Interval

    Returns
    -------
    numpy.array
    """
    return np.where(np.logical_and(interval == -1.0,point == 1.0),-1.0,point)

#Get interval as inf
def get_inf(point,interval):
    """
    Get interval [X,B] given interval [A,B] and point X in [A,B]
    -------
    Parameters
    ----------
    point : numpy.array

        Point

    interval : numpy.array

        Interval

    Returns
    -------
    numpy.array
    """
    return np.where(np.logical_and(interval == -1.0,point == 0.0),-1.0,point)

#Count the internal points in each interval
@jax.jit
def count_points(intervals):
    """
    Count internal points in each interval
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Intervals
    Returns
    -------
    jax.numpy.array
    """
    return jax.vmap(lambda interval: jnp.power(2,jnp.sum(interval == -1)))(intervals)

#Sample interval
def sample_break_interval(point_break,interval_break,rng):
    """
    Sample a break of an interval at an internal point
    -------
    Parameters
    ----------
    point_break : numpy.array

        Point to break interval on

    interval_break : numpy.array

        Which interval to break

    rng : numpy.random

        Random generator object

    Returns
    -------
    numpy.array
    """
    #Start new interval
    new_interval = point_break
    #Break inf or sup
    inf_sup = rng.choice(np.array([0,1]),size=(1,))
    if inf_sup == 0:
        new_interval = get_inf(new_interval,interval_break)
    else:
        new_interval = get_sup(new_interval,interval_break)
    return new_interval

#Update partition
def update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new):
    """
    Update partition after breaking interval
    -------
    Parameters
    ----------
    b_break : int

        Which block to break

    intervals : numpy.array

        Intervals of partition

    cover_intervals : numpy.array

        Intervals that cover [A,B]/[A,X] or [A,B]/[X,B]

    block : numpy.array

        Block of each interval

    index_interval : int

        Index of interval to break

    division_old : numpy.array

        Division of kept intervals into the two new blocks

    division_new : numpy.array

        Division of new intervals into the two new blocks

    Returns
    -------
    .numpy.arrays of intervals and block
    """
    intervals = np.append(intervals,cover_intervals,0)
    intervals = np.delete(intervals,index_interval,0)
    max_block = np.max(block)
    block = np.delete(block,index_interval)
    block = np.where(block == b_break,division_old,block)
    block = np.append(block,np.where(division_new == 0,b_break,max_block + 1))
    return intervals,block

#One step reduction
def reduce(intervals):
    """
    One step of intervals reduction
    -------
    Parameters
    ----------
    intervals : numpy.array

        Intervals of partition

    Returns
    -------
    numpy.array of intervals and logical indicating whether the returned intervals are reduced
    """
    for i in range(intervals.shape[0] - 1):
        for j in range(i + 1,intervals.shape[0]):
            if (np.where(intervals[i,:] == -1,1,0) == np.where(intervals[j,:] == -1,1,0)).all():
                if np.sum(intervals[i,:] != intervals[j,:]) == 1:
                    united = np.where(intervals[i,:] != intervals[j,:],-1.0,intervals[i,:])
                    intervals = np.delete(intervals,np.array([i,j]),0)
                    intervals = np.append(intervals,united.reshape((1,united.shape[0])),0)
                    return intervals,False
    return intervals,True

#Sample neighbor
def break_interval(point_break,index_interval,interval_break,b_break,intervals,block,nval,tab_train,tab_val,step,rng,num_classes = 2):
    """
    Break an interval on an internal point to get a neighbor
    -------
    Parameters
    ----------
    point_break : numpy.array

        Point to break interval on

    index_interval : numpy.array

        Index of interval to break

    interval_break : numpy.array

        Interval to break

    b_break : numpy.array

        Block which contains the break interval

    intervals : numpy.array

        Intervals

    block : numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    step : logical

        Whether to return the updated partition (take a step) or only its error

    rng : numpy.random

        Random generator object

    num_classses : int

        Number of classes

    Returns
    -------
    numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Sample interval
    new_interval = sample_break_interval(point_break,interval_break,rng)

    #Compute interval cover of complement
    where_fill = np.where(np.logical_and(new_interval != -1.0,interval_break == -1.0))[1]
    where_fill = rng.permutation(where_fill)
    cover_intervals = cover_break_interval(new_interval.copy(),where_fill)

    #Divide into two blocks
    division_new = np.append(np.array([1,0]),rng.choice(np.array([0,1]),size = (cover_intervals.shape[0] - 2,),replace = True))
    division_old = rng.choice(np.append(b_break,np.max(block) + 1),size = (block.shape[0] - 1,),replace = True)

    #Update partition
    intervals,block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes)

    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Unite two blocks
def unite_blocks(unite,intervals,block,nval,tab_train,tab_val,step,rng,num_classes = 2):
    """
    Unite blocks to obtain a neighbor
    -------
    Parameters
    ----------
    unite : jax.numpy.array

        Which blocks to unite

    intervals : jax.numpy.array

        Intervals

    block : jax.numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    step : logical

        Whether to return the updated partition (take a step) or only its error

    rng : numpy.random

        Random generator object

    num_classses : int

        Number of classes

    Returns
    -------
    numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Delete intervals and block united
    which_intervals = np.where(np.logical_or(block == unite[0],block == unite[1]))[0]
    unite_intervals = intervals[which_intervals,:]
    intervals = np.delete(intervals,which_intervals,0)
    block = np.delete(block,which_intervals)

    #Try to reduce united intervals
    reduced = False
    while(not reduced):
        unite_intervals,reduced = reduce(unite_intervals)

    #Update intervals and block
    intervals = np.append(intervals,unite_intervals,0)
    block = np.append(block,np.repeat(np.min(unite),unite_intervals.shape[0]))
    block = np.where(block > np.max(unite),block - 1,block)

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes)

    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Dismenber block
def dismenber_block(b_dis,intervals,block,nval,tab_train,tab_val,step,rng,num_classes = 2):
    """
    Dismenber block randomly to sample a neighbor
    -------
    Parameters
    ----------
    b_dis : int

        Which block to dismenber

    intervals : numpy.array

        Intervals

    block : numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train : numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    step : logical

        Whether to return the updated partition (take a step) or only its error

    rng : numpy.random

        Random generator object

    num_classses : int

        Number of classes

    Returns
    -------
    jax.numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #How to dismenber
    max_block = np.max(block) + 1
    division_new = rng.permutation(np.append(np.append(b_dis,max_block),rng.choice(np.append(b_dis,max_block),size = (np.sum(block == b_dis) - 2,),replace = True)))

    #Update block
    block[block == b_dis] = division_new

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes)
    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Frequency of labels of points in a interval
def frequency_labels_interval(interval,data,labels,num_classes = 2):
    return np.bincount(np.where(get_elements_interval(interval,data),labels,num_classes),length = num_classes + 1)[:-1]

frequency_labels_interval = jax.jit(frequency_labels_interval,static_argnames = ['num_classes'])

#Test if intervals are maximal
def is_maximal(intervals):
    """
    Test if intervals are maximal
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Intervals

    Returns
    -------
    jax.numpy.array of maximal intervals
    """
    for i in range(intervals.shape[0]):
        fixed = np.where(intervals[i,:] != -1)[0]
        free = np.where(intervals[i,:] == -1)[0]
        for j in range(intervals.shape[0]):
            if i != j and np.sum(intervals[j,] == -1) > 0:
                if free.shape[0] == 0:
                    if np.min(np.logical_or(intervals[j,fixed] == intervals[i,fixed],intervals[j,fixed] == -1)):
                        return False
                elif fixed.shape[0] == 0:
                    if np.min(intervals[j,free] == -1):
                        return False
                elif np.min(np.logical_or(intervals[j,fixed] == intervals[i,fixed],intervals[j,fixed] == -1)) and np.min(intervals[j,free] == -1):
                    return False
    return True

#Reduce to maximal intervals
def maximal(intervals):
    """
    Reduce to maximal intervals
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Intervals

    Returns
    -------
    jax.numpy.array of maximal intervals
    """
    delete = []
    for i in range(intervals.shape[0]):
        if contained_some(intervals[i,:],np.delete(intervals,i,0)):
            delete = delete + [i]
    if len(delete) > 0:
        intervals = np.delete(intervals,delete,0)
    return intervals

#Test if intervals contain/are contained
def contained(I1,I2):
    """
    Test if two intervals contain/are contained
    -------
    Parameters
    ----------
    I1,I2 : jax.numpy.array

        Intervals

    Returns
    -------
    logical
    """
    #I1 is contained in I2
    I2_fixedI1 = jnp.where(I1 != -1,I2,-1)
    I1_fixedI1 = jnp.where(I1 != -1,I1,-1)
    I2_freeI1 = jnp.where(I1 == -1,I2,-1)
    I1_freeI1 = jnp.where(I1 == -1,I1,-1)
    c1 = jnp.logical_and(jnp.min(jnp.logical_or(I2_fixedI1 == I1_fixedI1,I2_fixedI1 == -1)),jnp.min(I2_freeI1 == -1))
    #I2 is contained in I1
    tmp = I1.copy()
    I1 = I2
    I2 = tmp
    I2_fixedI1 = jnp.where(I1 != -1,I2,-1)
    I1_fixedI1 = jnp.where(I1 != -1,I1,-1)
    I2_freeI1 = jnp.where(I1 == -1,I2,-1)
    I1_freeI1 = jnp.where(I1 == -1,I1,-1)
    c2 = jnp.logical_and(jnp.min(jnp.logical_or(I2_fixedI1 == I1_fixedI1,I2_fixedI1 == -1)),jnp.min(I2_freeI1 == -1))
    return jnp.logical_or(c1,c2)

#Test if interval is contained/contain some interval
def contained_some(I1,intervals):
    """
    Test if an interval is contained in/contains some interval in a array
    -------
    Parameters
    ----------
    I1 : jax.numpy.array

        Interval

    intervals : jax.numpy.array

        Array of intervals

    Returns
    -------
    logical
    """
    return np.max(jax.vmap(lambda I2: contained(I1,I2))(intervals))
