#Utility functions lattDesc
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
import numpy as np
import time
from lattDesc import data as dt
import math

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

    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    train : jax.numpy.array

        Array with training data. The last column contains the labels

    val : jax.numpy.array

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
        jax.numpy.arrays of tab_train_batch and tab_val_batch and the batch validation sample size bnval
    """
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
        bnval = jnp.sum(tab_val_batch[:,-num_classes:])
    else: #No batch for validation
        tab_val_batch = tab_val #Copy frequency table
        bnval = nval #Copy validation sample size
    return tab_train_batch,tab_val_batch,bnval

#Teste partial order
def partial_order(x,y):
    """
    Test if x <= y
    -------
    Parameters
    ----------
    x : jax.numpy.array

        Point

    y : jax.numpy.array

        Point

    Returns
    -------
    logical
    """
    return jnp.sum(jnp.where(x <= y,1,0)) == x.shape[0]

#Test if element is in interval
@jax.jit
def test_interval(interval,x):
    """
    Test if element is in interval
    -------
    Parameters
    ----------
    interval : jax.numpy.array

        Interval

    x : jax.numpy.array

        Element

    Returns
    -------
    logical
    """
    fixed = jnp.where(interval >= 0,1,0)
    return jnp.sum(jnp.where(fixed == 1,x != interval,False)) == 0

#Test if element is not in interval
@jax.jit
def test_not_interval(interval,x):
    """
    Test if element is not in interval
    -------
    Parameters
    ----------
    interval : jax.numpy.array

        Interval

    x : jax.numpy.array

        Element

    Returns
    -------
    logical
    """
    fixed = jnp.where(interval >= 0,1,0)
    return jnp.sum(jnp.where(fixed == 1,x != interval,False)) != 0

#Teste if element is limit of interval (we know it is in interval)
@jax.jit
def test_limit_interval(interval,x):
    """
    Test if element is the limit of interval
    -------
    Parameters
    ----------
    interval : jax.numpy.array

        Interval

    x : jax.numpy.array

        Element

    Returns
    -------
    logical
    """
    x_max = jnp.where(interval < 0,x,-1)
    x_min = jnp.where(interval < 0,x,1)
    return jnp.max(x_max) == jnp.min(x_min)

#Get limits of interval
@jax.jit
def get_limits_interval(interval,data):
    """
    Flag the points in the data that are limits of interval
    -------
    Parameters
    ----------
    interval : jax.numpy.array

        Interval

    data : jax.numpy.array

        Data

    Returns
    -------
    jax.numpy.array of logical
    """
    return jax.vmap(lambda x: test_limit_interval(interval,x))(data)

#Get elements that are limit of some interval
@jax.jit
def get_limits_some_interval(intervals,data):
    """
    Flag the points in the data that are limits of some interval in a array
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Array of intervals

    data : jax.numpy.array

        Data

    Returns
    -------
    jax.numpy.array of logical
    """
    return jnp.sum(jax.vmap(lambda interval: get_limits_interval(interval,data))(intervals),0) > 0

#Get elements in interval
@jax.jit
def get_elements_interval(interval,data):
    """
    Flag the elements in dataset that are in an interval
    -------

    Parameters
    ----------
    interval : jax.numpy.array

        Interval

    data : jax.numpy.array

        Dataset

    Returns
    -------

    jax.numpy.array of logical

    """
    return jax.vmap(lambda x: test_interval(interval,x))(data)

#Get elements in each interval
@jax.jit
def get_elements_each_interval(intervals,data):
     return jax.vmap(lambda interval: get_elements_interval(interval,data))(intervals)

#Get elements in some interval
@jax.jit
def get_elements_some_interval(intervals,data):
    """
    Flag the elements in dataset that are in some interval in an array
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Array of intervals

    data : jax.numpy.array

        Data

    Returns
    -------
    jax.numpy.array of logical
    """
    return jnp.sum(jax.vmap(lambda interval: get_elements_interval(interval,data))(intervals),0) > 0

#Compute error of a block
def error_block(tab_train,tab_val,nval,key,num_classes = 2):
    """
    Compute the error a block
    -------
    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    nval : int

        Size of validation sample

    key : int

        Seed for random classification in the presence of ties

    num_classes : int

        Number of classes

    Returns
    -------
    float
    """
    freq = jnp.sum(tab_train[:,-num_classes:],0)
    pred = freq == jnp.max(freq)
    pred = jax.random.choice(jax.random.PRNGKey(key), jnp.arange(num_classes),shape=(1,),p = jnp.where(pred,1,0))
    freq_val = jnp.sum(tab_val[:,-num_classes:],0)
    err = jnp.sum(jnp.where(jnp.arange(num_classes) == pred,0,freq_val))
    return err/nval

error_block = jax.jit(error_block,static_argnames = ['num_classes'])

#Get label of each interval
def estimate_label_block(tab_train,intervals,num_classes = 2,key = 0):
    """
    Estimate the label of an interval from training data
    -------
    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : jax.numpy.array

        Intervals of block

    num_classes : int

        Number of classes

    key : int

        Seed for random classification in the presence of ties

    Returns
    -------
    jax.numpy.array
    """
    tab_train = ftable_block(intervals,tab_train,num_classes)
    freq = jnp.sum(tab_train[:,-num_classes:],0)
    pred = freq == jnp.max(freq)
    pred = jax.random.choice(jax.random.PRNGKey(key), jnp.arange(num_classes),shape=(1,),p = jnp.where(pred,1,0))
    return pred

estimate_label_block = jax.jit(estimate_label_block,static_argnames = ['num_classes'])

#Estimate the label of a partition
def estimate_label_partition(tab_train,intervals,block,num_classes = 2,key = 0):
    """
    Estimate the label of a partition from training data
    -------
    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : jax.numpy.array

        Array of intervals that define the partition

    block : jax.numpy.array

        Array with the block of each interval

    num_classes : int

        Number of classes

    key : int

        Seed for random classification in the presence of ties

    Returns
    -------
    jax.numpy.array
    """
    pred = jnp.zeros((intervals.shape[0],))
    for i in range(jnp.max(block) + 1):
        tmp_intervals = intervals[block == i,:]
        pred_block = estimate_label_block(tab_train,tmp_intervals,num_classes,key)
        pred = jnp.where(block == i,pred_block,pred)
    return pred

#Get estimated function
def get_estimated_function(tab_train,intervals,block,num_classes = 2,key = 0):
    """
    Get estimated function of partition
    -------
    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : jax.numpy.array

        Array of intervals that define the partition

    block : jax.numpy.array

        Array with the block of each interval

    num_classes : int

        Number of classes

    key : int

        Seed for random classification in the presence of ties

    Returns
    -------
    jax.numpy.array
    """
    label_blocks = estimate_label_partition(tab_train,intervals,block,num_classes,key)
    def f(point):
        return jnp.sum(jnp.where(get_interval(point,intervals),label_blocks,0))
    return jax.jit(f)

#Frequency table of block
def ftable_block(intervals,tab,num_classes = 2):
    """
    Frequency table of a block
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Array of intervals that define the block

    tab : jax.numpy.array

        Frequency table. Each row refers to an input point and the num_classes columns refer to label frequencies

    num_classes : int

        Number of classes

    Returns
    -------
    jax.numpy.array
    """
    return jax.lax.select(jnp.repeat(get_elements_some_interval(intervals,tab[:,0:-num_classes,]).reshape((tab.shape[0],1)),tab.shape[1],1),tab,jnp.zeros(tab.shape).astype(tab.dtype))

ftable_block = jax.jit(ftable_block,static_argnames = ['num_classes'])

#Get error partition (it is better to not jit)
def error_partition(tab_train,tab_val,intervals,block,nval,key,num_classes = 2):
    """
    Get error of partition
    -------
    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last num_classes columns refer to label frequencies

    intervals : jax.numpy.array

        Array of intervals that define the partition

    block : jax.numpy.array

        Array with the block of each interval

    nval : int

        Size of validation sample

    key : int

        Seed for random classification in the presence of ties

    num_classes : int

        Number of classes

    Returns
    -------
    float
    """
    error = 0
    for i in range(jnp.max(block) + 1):
        tmp_intervals = intervals[block == i,:]
        tab_train_block = ftable_block(tmp_intervals,tab_train,num_classes)
        tab_val_block = ftable_block(tmp_intervals,tab_val,num_classes)
        error = error + error_block(tab_train_block,tab_val_block,nval,key,num_classes)
    return error

#Break interval at new interval
@jax.jit
def cover_break_interval(new_interval,where_fill):
    """
    Compute a cover of [A,B]/[A,X] or [A,B]/[X,B] by intervals
    -------
    Parameters
    ----------
    new_interval : jax.numpy.array

        The interval [A,X] or [X,B]

    where_fill : jax.numpy.array

        Pre-computed random ordering of the free positions of [A,B] that are not free in new_interval

    Returns
    -------
    jax.numpy.array of intervals
    """
    cover_intervals = None
    for i in range(len(where_fill)):
        tmp = new_interval
        tmp = tmp.at[0,where_fill[i]].set(1 - tmp[0,where_fill[i]])
        tmp = tmp.at[0,where_fill[(i+1):]].set(-1)
        if cover_intervals is None:
            cover_intervals = tmp
        else:
            cover_intervals = jnp.append(cover_intervals,tmp,0)
    cover_intervals = jnp.append(cover_intervals,new_interval,0)
    return cover_intervals

#Get interval as sup
@jax.jit
def sample_sup(point,interval):
    """
    Get interval [A,X] given interval [A,B] and point X in [A,B]
    -------
    Parameters
    ----------
    point : jax.numpy.array

        Point

    interval : jax.numpy.array

        Interval

    Returns
    -------
    jax.numpy.array
    """
    return jnp.where((interval == -1.0)*(point == 1.0),-1.0,point)

#Get interval as inf
@jax.jit
def sample_inf(point,interval):
    """
    Get interval [X,B] given interval [A,B] and point X in [A,B]
    -------
    Parameters
    ----------
    point : jax.numpy.array

        Point

    interval : jax.numpy.array

        Interval

    Returns
    -------
    jax.numpy.array
    """
    return jnp.where((interval == -1.0)*(point == 0.0),-1.0,point)

#Flag intervals that contain point
@jax.jit
def get_interval(point,intervals):
    """
    Flag interval that constains a point
    -------
    Parameters
    ----------
    point : jax.numpy.array

        Point

    intervals : jax.numpy.array

        Intervals

    Returns
    -------
    jax.numpy.array of logical
    """
    return jax.vmap(lambda interval: test_interval(interval,point))(intervals)

#Sample interval
def sample_break_interval(point,intervals,which_interval,domain,key):
    """
    Sample interval to break at a given point
    -------
    Parameters
    ----------
    point : jax.numpy.array

        Point

    interval : jax.numpy.array

        Intervals

    key : int

        Seed for sampling

    Returns
    -------
    jax.numpy.array
    """
    #Get interval to break on
    break_interval = intervals[which_interval,:]
    if test_limit_interval(break_interval,point):
        if jnp.sum(break_interval == -1) == 1:
            return None,None
        points_interval = domain[get_limits_some_interval(break_interval,domain),:]
        points_interval = jit_row_delete(points_interval,jnp.where(get_limits_some_interval(break_interval,points_interval))[0])
        if points_interval.shape[0] == 0:
            value = jnp.append(jnp.arange(2),jax.random.choice(jax.random.PRNGKey(key+1),jnp.arange(2),shape = (1,jnp.sum(break_interval == -1) - 2)))
            value = jax.random.permutation(jax.random.PRNGKey(key + 2),value,0)
            value = jnp.zeros((1,break_interval.shape[1])).at[jnp.where(break_interval == -1)].set(value)
            new_interval = break_interval.at[0,:].set(jnp.where(break_interval == -1,value,break_interval)[0,:])
        else:
            new_interval = points_interval[jax.random.choice(jax.random.PRNGKey(key), jnp.arange(points_interval.shape[0]),shape=(1,)),:]
        if partial_order(new_interval,point):
            new_interval = sample_sup(new_interval,break_interval)
        else:
            new_interval = sample_inf(new_interval,break_interval)
        return new_interval,break_interval
    else:
        #Break inf or sup
        new_interval = point
        inf_sup = jax.random.choice(jax.random.PRNGKey(key), jnp.array([0,1]),shape=(1,))
        if inf_sup == 0:
            new_interval = sample_inf(new_interval,break_interval)
        else:
            new_interval = sample_sup(new_interval,break_interval)
        return new_interval,break_interval

#Update partition
def update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new):
    """
    Update partition after breaking interval
    -------
    Parameters
    ----------
    b_break : int

        Which block to break

    intervals : jax.numpy.array

        Intervals of partition

    cover_intervals : jax.numpy.array

        Intervals that cover [A,B]/[A,X] or [A,B]/[X,B]

    block : jax.numpy.array

        Block of each interval

    index_interval : int

        Index of interval to break

    division_old : jax.numpy.array

        Division of kept intervals into the two new blocks

    division_new : jax.numpy.array

        Division of new intervals into the two new blocks

    Returns
    -------
    jax.numpy.arrays of intervals and block
    """
    intervals = jnp.append(intervals,cover_intervals,0)
    intervals = jit_row_delete(intervals,index_interval)
    max_block = jnp.max(block)
    block = jit_delete(block,index_interval)
    block = jnp.where(block == b_break,division_old,block)
    block = jnp.append(block,jnp.where(division_new == 0,b_break,max_block + 1))
    return intervals,block


#One step reduction
def reduce(intervals):
    """
    One step of intervals reduction
    -------
    Parameters
    ----------
    intervals : jax.numpy.array

        Intervals of partition

    Returns
    -------
    jax.numpy.array of intervals and logical indicating whether the returned intervals are reduced
    """
    for i in range(intervals.shape[0] - 1):
        for j in range(i + 1,intervals.shape[0]):
            if (jnp.where(intervals[i,:] == -1,1,0) == jnp.where(intervals[j,:] == -1,1,0)).all():
                if jnp.sum(intervals[i,:] != intervals[j,:]) == 1:
                    united = jnp.where(intervals[i,:] != intervals[j,:],-1.0,intervals[i,:])
                    intervals = jnp.delete(intervals,jnp.array([i,j]),0)
                    intervals = jnp.append(intervals,united.reshape((1,united.shape[0])),0)
                    return intervals,False
    return intervals,True

#Jit delete
@jax.jit
def jit_delete(x, i):
    """
    Jitted delete function
    -------
    Parameters
    ----------
    x : jax.numpy.array

        1D array

    i : jax.numpy.array

        Indexes to delete

    Returns
    -------
    jax.numpy.array
    """
    return jnp.delete(x,i,assume_unique_indices = True)

@jax.jit
def jit_row_delete(x, i):
    """
    Jitted row delete function
    -------
    Parameters
    ----------
    x : jax.numpy.array

    21D array

    i : jax.numpy.array

        Indexes to delete

    Returns
    -------
    jax.numpy.array
    """
    return jnp.delete(x,i,0,assume_unique_indices = True)

#Sample neighbor
def break_interval(point_break,intervals,block,nval,tab_train,tab_val,step,key,num_classes = 2):
    """
    Break an interval randomly to sample a neighbor
    -------
    Parameters
    ----------
    point_break : jax.numpy.array

        Which point to break on

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

    key : int

        Seed for sampling

    num_classses : int

        Number of classes

    Returns
    -------
    jax.numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),10)

    #Sample interval
    t0 = time.time()
    which_interval = get_interval(point_break[:,:-num_classes],intervals)
    new_interval,break_interval = sample_break_interval(point_break[:,:-num_classes],intervals,which_interval,tab_train[:,:-num_classes],key[0,0])
    if new_interval is None:
        if not step:
            return 1.1
        else:
            return {'block': block,'intervals': intervals,'error': 1.1}
    index_interval = jnp.where(which_interval)[0]
    b_break = block[index_interval]


    #Compute interval cover of complement
    where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
    where_fill = jax.random.permutation(jax.random.PRNGKey(key[1,0]), where_fill)
    cover_intervals = cover_break_interval(new_interval,where_fill)

    #Divide into two blocks
    t1 = time.time()
    division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0] - 2,),replace = True))
    division_old = jax.random.choice(jax.random.PRNGKey(key[3,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (block.shape[0] - 1,),replace = True)

    #Update partition
    intervals,block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,key[4,0],num_classes)

    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Unite two blocks
def unite_blocks(unite,intervals,block,nval,tab_train,tab_val,step,key,num_classes = 2):
    """
    Unite blocks randomly to sample a neighbor
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

    key : int

        Seed for sampling

    num_classses : int

        Number of classes

    Returns
    -------
    jax.numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),10)

    #Delete intervals and block united
    which_intervals = jnp.where(jnp.logical_or(block == unite[0],block == unite[1]))[0]
    unite_intervals = intervals[which_intervals,:]
    intervals = jit_row_delete(intervals,which_intervals)
    block = jit_delete(block,which_intervals)

    #Try to reduce united intervals
    t0 = time.time()
    reduced = False
    while(not reduced):
        unite_intervals,reduced = reduce(unite_intervals)

    #Update intervals and block
    intervals = jnp.append(intervals,unite_intervals,0)
    block = jnp.append(block,jnp.repeat(jnp.min(unite),unite_intervals.shape[0]))
    block = jnp.where(block > jnp.max(unite),block - 1,block)

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,key[0,0],num_classes)

    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Dismenber block
def dismenber_block(b_dis,intervals,block,nval,tab_train,tab_val,step,key,num_classes = 2):
    """
    Dismenber block randomly to sample a neighbor
    -------
    Parameters
    ----------
    b_dis : int

        Which block to dismenber

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

    key : int

        Seed for sampling

    num_classses : int

        Number of classes

    Returns
    -------
    jax.numpy.arrays of intervals and block and/or the error of the sampled partition
    """
    #Seed
    t0 = time.time()
    key = jax.random.split(jax.random.PRNGKey(key),10)

    #Test if can be dismenbered (get number of elements in each interval)
    occupied = jnp.sum(get_elements_each_interval(intervals[block == b_dis,:],tab_train[:,:-num_classes]),1) > 0
    if jnp.sum(occupied) < 2:
        return {'block': block,'intervals': intervals,'error': 1.1}

    #How to dismenber
    max_block = jnp.max(block) + 1
    sample_div_occupied = np.append(jnp.append(b_dis,max_block),jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.append(b_dis,max_block),shape = (jnp.sum(block == b_dis) - 2,),replace = True))
    sample_div_empty = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.append(b_dis,max_block),shape = (jnp.sum(block == b_dis),),replace = True)
    division_new = jnp.where(occupied,sample_div_occupied,sample_div_empty)

    #Update block
    block = block.at[block == b_dis].set(division_new)

    #Compute error
    error = error_partition(tab_train,tab_val,intervals,block,nval,key[3,0],num_classes)
    #Return
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Frequency of labels of points in a interval
def frequency_labels_interval(interval,data,labels,num_classes = 2):
    return jnp.bincount(jnp.where(get_elements_interval(interval,data),labels,num_classes),length = num_classes + 1)[:-1]

frequency_labels_interval = jax.jit(frequency_labels_interval,static_argnames = ['num_classes'])
