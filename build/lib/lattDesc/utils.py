#Utility functions lattDesc
import numpy as np
import time
from lattDesc import data as dt
from lattDesc import learning as ipl
import math
import pickle

#Get frequency table of batch
def get_tfrequency_batch(tab_train,tab_val,bsize,bsize_val,batch_val,nval,rng,num_classes = 2):
    """
    Sample the frequency table of a batch
    -------
    Parameters
    ----------
    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

    bsize : int

        Batch size of training

    bsize_val : int

        Batch size of validation

    batch_val : logical

        Whether to consider batches for the validation data

    nval : int

        Size of validation sample

    Returns
    -------
    numpy.arrays of tab_train,tab_train_batch,tab_val,tab_val_batch and the batch validation sample size bnval
    """
    #Sample points in training sample that will be in the batch
    ids = np.sort(rng.choice(np.arange(np.sum(tab_train[:,-num_classes:])),size = (bsize,1),replace = False),0)
    #Find the sampled points in the frequencey table
    vector_tab_train = tab_train[:,-num_classes:].reshape((tab_train.shape[0]*num_classes))
    pos_ids = np.apply_along_axis(lambda id: np.min(np.where(np.cumsum(vector_tab_train) >= id)[0]),1,ids)
    #Build batch frequency table
    tab_train_batch = np.append(tab_train[:,:-num_classes],np.bincount(pos_ids,minlength = vector_tab_train.shape[0]).reshape((tab_train.shape[0],num_classes)),1)
    #Erase sample points from the training table so they cannot be sampled in another batch
    tab_train[:,-num_classes:] = tab_train[:,-num_classes:] - tab_train_batch[:,-num_classes:]
    #Keep only points that appear in the batch
    tab_train_batch = tab_train_batch[np.sum(tab_train_batch[:,-num_classes:],1) > 0,:]

    #Sample points validation sample
    if batch_val:
        #Sample points in validation sample that will be in the batch
        ids = np.sort(rng.choice(np.arange(np.sum(tab_val[:,-num_classes:])),size = (bsize_val,1),replace = False),0)
        #Find the sampled points in the frequencey table
        vector_tab_val = tab_val[:,-num_classes:].reshape((tab_val.shape[0]*num_classes))
        pos_ids = np.apply_along_axis(lambda id: np.min(np.where(np.cumsum(vector_tab_val) >= id)[0]),1,ids)
        #Build batch frequency table
        tab_val_batch = np.append(tab_val[:,:-num_classes],np.bincount(pos_ids,minlength = vector_tab_val.shape[0]).reshape((tab_val.shape[0],num_classes)),1)
        #Erase sample points from the validation table so they cannot be sampled in another batch
        tab_val[:,-num_classes:] = tab_val[:,-num_classes:] - tab_val_batch[:,-num_classes:]
        #Keep only points that appear in the batch
        tab_val_batch = tab_val_batch[np.sum(tab_val_batch[:,-num_classes:],1) > 0,:]
        #Update size of validation sample
        bnval = np.sum(tab_val_batch[:,-num_classes:])
    else:
        tab_val_batch = tab_val
        bnval = nval

    return tab_train,tab_train_batch,tab_val,tab_val_batch,bnval

#Teste partial order
def partial_order(x,y):
    """
    Test if x <= y for x and y in a Boolean lattice
    -------
    Parameters
    ----------
    x,y : numpy.array

        Points in a Boolean lattice

    Returns
    -------
    logical
    """
    return np.sum(x <= y) == x.shape[0]

#Test if element is in interval
def test_interval(interval,x):
    """
    Test if element x is in interval
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
    Test if element x is not in interval
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

#Test if element is limit of interval (assume it is in interval)
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
    Flag the points in dataset that are limits of interval
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
    return np.apply_along_axis(lambda x: test_limit_interval(interval,x),1,data)

#Get limits of each interval
def get_limits_each_interval(intervals,data):
    """
    Flag the elements in dataset that are the limits of each interval
    -------
    Parameters
    ----------
    intervals : numpy.array

        Array of intervals

    data : numpy.array

        Dataset

    Returns
    -------
    numpy.array of logical
    """
    return np.apply_along_axis(lambda interval: get_limits_interval(interval,data),1,intervals)

#Get elements that are limit of some interval
def get_limits_some_interval(intervals,data):
    """
    Flag the points in dataset that are limits of some interval in a array
    -------
    Parameters
    ----------
    intervals : numpy.array

        Array of intervals

    data : numpy.array

        Dataset

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

        Array of intervals

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

        Array of intervals

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

        Dataset

    Returns
    -------
    numpy.array of logical
    """
    return np.sum(get_elements_each_interval(intervals,data),0) > 0

#Compute error
def val_error(tab_train,tab_val,nval,rng,num_classes = 2):
    """
    Compute the validation error for given training and validation frequency tables
    -------
    Parameters
    ----------
    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

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
    p = np.where(freq == np.max(freq),1,0)
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

        Array with a frequency table

    num_classes : int

        Number of classes

    Returns
    -------
    numpy.array
    """
    tab_block = tab[get_elements_some_interval(intervals,tab[:,:-num_classes,]),:]
    return np.sum(tab_block[:,-num_classes:],0)

#Get label of each interval
def estimate_label_block(tab_train,intervals,rng,num_classes = 2):
    """
    Estimate the label of an interval from training data
    -------
    Parameters
    ----------
    tab_train : numpy.array

        Array with the frequency table of training

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
    p = np.where(tab_train == np.max(tab_train),1,0)
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

        Array with the frequency table of training

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

        Array with the frequency table of training

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
    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

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
    return np.where(np.logical_and(interval == -1,point == 1),-1,point)

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
    return np.where(np.logical_and(interval == -1,point == 0),-1,point)

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
    numpy.arrays of intervals and block
    """
    intervals = np.append(intervals,cover_intervals,0) #Add new intervals
    intervals = np.delete(intervals,index_interval,0) #Delete broken interval
    max_block = np.max(block) #Maximum block
    block = np.delete(block,index_interval) #Delete block of broken interval
    block = np.where(block == b_break,division_old,block) #New block of the kept intervals
    block = np.append(block,np.where(division_new == 0,b_break,max_block + 1)) #Block of the new intervals
    return intervals,block

#One step reduction
def reduce(intervals):
    """
    One step of intervals reduction
    -------
    Parameters
    ----------
    intervals : numpy.array

        Intervals of a block

    Returns
    -------
    numpy.array of intervals and logical indicating whether the returned intervals are reduced
    """
    for i in range(intervals.shape[0] - 1):
        for j in range(i + 1,intervals.shape[0]):
            if (np.where(intervals[i,:] == -1,1,0) == np.where(intervals[j,:] == -1,1,0)).all():
                if np.sum(intervals[i,:] != intervals[j,:]) == 1:
                    united = np.where(intervals[i,:] != intervals[j,:],-1,intervals[i,:])
                    intervals = np.delete(intervals,np.array([i,j]),0)
                    intervals = np.append(intervals,united.reshape((1,united.shape[0])),0)
                    return intervals,False
    return intervals,True

#Sample neighbor by breaking interval
def break_interval(point_break,index_interval,interval_break,b_break,intervals,block,nval,tab_train,tab_val,step,rng,num_classes = 2,compute_error = True):
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

        Array of intervals

    block : numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

    step : logical

        Whether to return the updated partition (take a step) or only its error

    rng : numpy.random

        Random generator object

    num_classses : int

        Number of classes

    compute_error : logical

        Whether to compute the updated partition error

    Returns
    -------
    numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Sample interval
    new_interval = sample_break_interval(point_break,interval_break,rng)

    #Compute interval cover of complement
    where_fill = np.where(np.logical_and(new_interval != -1,interval_break == -1))[1]
    where_fill = rng.permutation(where_fill)
    cover_intervals = cover_break_interval(new_interval.copy(),where_fill)

    #Divide into two blocks
    division_new = np.append(np.array([1,0]),rng.choice(np.array([0,1]),size = (cover_intervals.shape[0] - 2,),replace = True))
    division_old = rng.choice(np.append(b_break,np.max(block) + 1),size = (block.shape[0] - 1,),replace = True)

    #Update partition
    intervals,block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)

    #Compute error
    if compute_error:
        error = error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes)
    else:
        error = None

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
    unite : numpy.array

        Which blocks to unite

    intervals : numpy.array

        Array of intervals

    block : numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

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

        Array of intervals

    block : numpy.array

        Block of each interval

    nval : int

        Validation sample size

    tab_train,tab_val : numpy.array

        Array with the frequency table of training and validation data

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

#Sample a neighbor and visit
def sample_visit_neighbor(kind_nei,prob_dismember,prob_break,block,intervals,bnval,tab_train_batch,tab_val_batch,rng,num_classes):
    if kind_nei == 0: #Unite intervals
        #Sample block to unite intervals
        unite = rng.choice(np.arange(np.max(block) + 1),size=(2,),replace = False)
        #Sample intervals to unite in the sampled block and return the result
        return unite_blocks(unite,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,rng = rng,num_classes = num_classes)
    elif kind_nei == 1: #Dismenber intervals
        #Sample block to dismenber intervals
        b_dis = rng.choice(np.arange(np.max(block) + 1),size=(1,),p = prob_dismember)
        #Sample dismenbering of the sampled block and return the result
        return dismenber_block(b_dis,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,rng = rng,num_classes = num_classes)
    elif kind_nei == 2: #Break interval
        #Sample point to break on
        point_break = tab_train_batch[rng.choice(np.arange(tab_train_batch.shape[0]),size = (1,),p = prob_break),:-num_classes][0,:]
        interval_index = np.where(get_interval(point_break,intervals))[0]
        #Break interval on sampled point and return the result
        return break_interval(point_break,interval_index,intervals[interval_index,:].copy(),block[interval_index].copy(),intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,rng = rng,num_classes = num_classes)

#Test if intervals contain/are contained
def contained(I1,I2):
    """
    Test if two intervals contain/are contained
    -------
    Parameters
    ----------
    I1,I2 : numpy.array

        Intervals

    Returns
    -------
    logical
    """
    #I1 is contained in I2
    c1 = np.sum(I2[I2 != -1] == I1[I2 != -1]) == np.sum(I2 != -1)
    #I2 is contained in I1
    c2 = np.sum(I1[I1 != -1] == I2[I1 != -1]) == np.sum(I1 != -1)
    return np.logical_or(c1,c2)

#Test if interval is contained/contain some interval
def contained_some(I1,intervals):
    """
    Test if an interval is contained in/contains some interval in a array
    -------
    Parameters
    ----------
    I1 : numpy.array

        Interval

    intervals : numpy.array

        Array of intervals

    Returns
    -------
    logical
    """
    return np.max(np.apply_along_axis(lambda I2: contained(I1,I2),1,intervals))

def run_experiment(tab_train,tab_val,tab_test,tab_tv,batches,sample,epochs,path_sufix,ISI = True):
    #Train by ERM
    print('ERM')
    res_erm = ipl.ERM(tab_train = tab_train,tab_test = tab_test)
    print(res_erm['test_error'])
    print('ERM tv')
    res_erm_tv = ipl.ERM(tab_train = tab_tv,tab_test = tab_test)
    print(res_erm_tv['test_error'])
    res_erm['f'] = None
    res_erm_tv['f'] = None
    with open('results/ERM_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_erm, f)
    with open('results/ERM_tv_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_erm_tv, f)

    #Train with ISI
    if ISI:
        res_ISI = ipl.ISI(tab_train = tab_train,tab_test = tab_test)
        print(res_ISI['test_error'])
        res_ISI_tv = ipl.ISI(tab_train = tab_tv,tab_test = tab_test)
        print(res_ISI_tv['test_error'])
        res_ISI['f'] = None
        res_ISI_tv['f'] = None
        with open('results/ISI_' + path_sufix + '.pkl', 'wb') as f:
            pickle.dump(res_ISI, f)
        with open('results/ISI_tv_' + path_sufix + '.pkl', 'wb') as f:
            pickle.dump(res_ISI_tv, f)

    #Lattice descent starting from unitary partition
    res_LDA = ipl.sdesc_BIPL(tab_train = tab_train,tab_val = tab_val,tab_test = tab_test,epochs = epochs,sample = sample,key = 0,batches = batches,num_classes = 2,n_jobs = sample)
    print(res_LDA['test_error'])
    res_LDA['f'] = None
    with open('results/LDA_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_LDA, f)

    #Lattice descent starting from unitary partition
    res_LDA_tv = ipl.sdesc_BIPL(tab_train = tab_tv,tab_val = tab_tv,tab_test = tab_test,epochs = epochs,sample = sample,key = 0,batches = batches,num_classes = 2,n_jobs = sample)
    print(res_LDA_tv['test_error'])
    res_LDA_tv['f'] = None
    with open('results/LDA_tv_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_LDA_tv, f)

    #Lattice descent with pre-training
    init = ipl.disjoint_ISI(tab_train = tab_train)
    res_LDAPre = ipl.sdesc_BIPL(intervals = init['intervals'],block = init['block'],tab_train = tab_train,tab_val = tab_val,tab_test = tab_test,epochs = epochs,sample = sample,key = 0,batches = batches,num_classes = 2,n_jobs = sample)
    print(res_LDAPre['test_error'])
    init_tv = ipl.disjoint_ISI(tab_train = tab_tv)
    res_LDAPre_tv = ipl.sdesc_BIPL(intervals = init_tv['intervals'],block = init_tv['block'],tab_train = tab_tv,tab_val = tab_tv,tab_test = tab_test,epochs = epochs,sample = sample,key = 0,batches = batches,num_classes = 2,n_jobs = sample)
    print(res_LDAPre_tv['test_error'])
    res_LDAPre['f'] = None
    res_LDAPre_tv['f'] = None
    init['f'] = None
    init_tv['f'] = None
    with open('results/LDAPre_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_LDAPre, f)
    with open('results/LDAPre_tv_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(res_LDAPre_tv, f)
    with open('results/init_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(init, f)
    with open('results/init_tv_' + path_sufix + '.pkl', 'wb') as f:
        pickle.dump(init_tv, f)
