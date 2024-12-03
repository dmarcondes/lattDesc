#Utility functions lattDesc
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
import numpy as np
import time
from lattDesc import data as dt

#Get frequency table of batch
def get_tfrequency_batch(b,batches,tab_train,tab_val,train,val,bsize,bsize_val,unique,batch_val):
    if b < batches - 1: #If it is not last batch
        if unique: #If data is unique, batch of frequency table
            tab_train_batch = tab_train[((b-1)*bsize):(b*bsize),:] #Get frequency table of batch
        else: #Else, compute frequency table of data batch
            tab_train_batch = dt.get_ftable(train[((b-1)*bsize):(b*bsize),:],unique) #Compute frequency table of batch
    else: #For the last batch
        if unique: #If data is unique, batch of frequency table
            tab_train_batch = tab_train[((b-1)*bsize):,:] #Get frequency table of batch
        else: #Else, compute frequency table of data batch
            tab_train_batch = dt.get_ftable(train[((b-1)*bsize):,:],unique) #Compute frequency table of batch
    if batch_val: #If batches for validation should be considered
        if b < batches - 1: #If it is not last batch
            if unique: #If data is unique, batch of frequency table
                tab_val_batch = val_train[((b-1)*bsize_val):(b*bsize_val),:] #Get frequency table of batch
            else: #Else, compute frequency table of data batch
                tab_val_batch = dt.get_ftable(val[((b-1)*bsize_val):(b*bsize_val),:],unique) #Compute frequency table of batch
        else: #For the last batch
            if unique: #If data is unique, batch of frequency table
                tab_val_batch = tab_val[((b-1)*bsize_val):,:] #Get frequency table of batch
            else: #Else, compute frequency table of data batch
                tab_val_batch = dt.get_ftable(val[((b-1)*bsize_val):,:],unique) #Compute frequency table of batch
        bnval = jnp.sum(tab_val_batch[:,-2:])
    else: #No batch for validation
        tab_val_batch = tab_val #Copy frequency table
        bnval = nval #Copy validation sample size
    return tab_train_batch,tab_val_batch,nbval

#Test if element in interval
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

#Test if element not in interval
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
    Test if element is the limit of an interval
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
    Flag the points in the data that are limits of the interval
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

        Dataset

    Returns
    -------

    jax.numpy.array of logical

    """
    return jnp.sum(jax.vmap(lambda interval: get_elements_interval(interval,data))(intervals),0) > 0

#Compute error of a block
@jax.jit
def error_block_partition(tab_train,tab_val,nval,key):
    """
    Compute the error a block
    -------

    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last two columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last two columns refer to label frequencies

    nval : int

        Size of validation sample

    key : int

        Key for random classification in the presence of ties

    Returns
    -------

    float

    """
    freq = jnp.sum(tab_train[:,-2:],0)
    pred = jnp.where(freq == jnp.max(freq),False,True)
    pred = pred.at[0].set(jax.random.choice(jax.random.PRNGKey(key), jnp.array([False,True]),shape=(1,),p = 1 - pred)[0])
    pred = pred.at[1].set(jax.random.choice(jax.random.PRNGKey(key+1), jnp.array([False,True]),shape=(1,),p = jnp.append(pred[0],1 - pred[0]))[0])
    freq_val = jnp.sum(tab_val[:,-2:],0)
    err = jnp.where(pred,freq_val,0)
    return jnp.sum(err)/nval
    #Estimate class
    #freq = jnp.sum(tab_train[:,-2:],0)
    #if freq[1] > freq[0]:
    #       not_pred = 0
    #elif freq[0] < freq[1]:
    #    not_pred = 1
    #else:
    #    not_pred = jax.random.choice(jax.random.PRNGKey(key), jnp.array([0,1]),shape=(1,))
    #freq_val = jnp.sum(tab_val[:,-2:],0)
    #return freq_val[not_pred]/nval

#Frequency table of block
@jax.jit
def ftable_block(intervals,tab):
    """
    Frequency table of a block
    -------

    Parameters
    ----------
    intervals : jax.numpy.array

        Array of intervals that define the block

    tab : jax.numpy.array

        Frequency table. Each row refers to an input point and the last two columns refer to label frequencies

    Returns
    -------

    jax.numpy.array

    """
    return jax.lax.select(jnp.repeat(get_elements_some_interval(intervals,tab[:,0:-2,]).reshape((tab.shape[0],1)),tab.shape[1],1),tab,jnp.zeros(tab.shape).astype(tab.dtype)) #tab[get_elements_some_interval(intervals,tab[:,0:-2,]),:]

#Get error partition (it is better to not jit)
def get_error_partition(tab_train,tab_val,intervals,block,nval,key):
    """
    Get error of partition
    -------

    Parameters
    ----------
    tab_train : jax.numpy.array

        Frequency table of training data. Each row refers to an input point and the last two columns refer to label frequencies

    tab_val : jax.numpy.array

        Frequency table of validation data. Each row refers to an input point and the last two columns refer to label frequencies

    intervals : jax.numpy.array

        Array of intervals that define the partition.

    block : jax.numpy.array

        Array with the block of each interval.

    nval : int

        Size of validation sample

    key : int

        Key for random classification in the presence of ties

    Returns
    -------

    float

    """
    error = 0
    for i in range(jnp.max(block) + 1):
        tmp_intervals = intervals[block == i,:]
        tab_train_block = ftable_block(tmp_intervals,tab_train)
        tab_val_block = ftable_block(tmp_intervals,tab_val)
        error = error + error_block_partition(tab_train_block,tab_val_block,nval,key)
    return error

#Break interval at new interval
def cover_break_interval(new_interval,where_fill):
    """
    Compute a cover of [A,B]/[A,X] or [A,B]/[X,B] by intervals.
    -------

    Parameters
    ----------
    new_interval : jax.numpy.array

        The interval [A,X] or [X,B].

    where_fill : jax.numpy.array

        Pre-computed random ordering of the free positions of [A,B] that are not free in new_interval.

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
    Get interval contained in an interval that ends in a point in it
    -------

    Parameters
    ----------
    point : jax.numpy.array

        Point.

    interval : jax.numpy.array

        Interval.

    Returns
    -------

    jax.numpy.array

    """
    return jnp.where((interval == -1.0)*(point == 1.0),-1.0,point)

#Get interval as inf
@jax.jit
def sample_inf(point,interval):
    """
    Get interval contained in an interval that starts in a point in it
    -------

    Parameters
    ----------
    point : jax.numpy.array

        Point.

    interval : jax.numpy.array

        Interval.

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

    interval : jax.numpy.array

        Intervals

    Returns
    -------

    jax.numpy.array

    """
    return jax.vmap(lambda interval: test_interval(interval,point))(intervals)

#Sample interval
def get_break_interval(point,intervals,which_interval,key):
    """
    Get random interval to break at a given point
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
    #Sample an interval in this block to break on
    break_interval = intervals[which_interval,:]
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

        Intervals.

    key : int

        Seed for sampling.

    Returns
    -------

    jax.numpy.array

    """
    intervals = jnp.append(intervals,cover_intervals,0)
    intervals = jnp.delete(intervals,index_interval,0)
    max_block = jnp.max(block)
    block = jnp.delete(block,index_interval)
    block = block.at[block == b_break].set(division_old)
    block = jnp.append(block,jnp.where(division_new == 0,b_break,max_block + 1))
    return intervals,block

#One step reduction
def reduce(intervals):
    for i in range(intervals.shape[0] - 1):
        for j in range(i + 1,intervals.shape[0]):
            if (jnp.where(intervals[i,:] == -1,1,0) == jnp.where(intervals[j,:] == -1,1,0)).all():
                if jnp.sum(intervals[i,:] != intervals[j,:]) == 1:
                    united = jnp.where(intervals[i,:] != intervals[j,:],-1.0,intervals[i,:])
                    intervals = jnp.delete(intervals,jnp.array([i,j]),0)
                    intervals = jnp.append(intervals,united.reshape((1,united.shape[0])),0)
                    return intervals,False
    return intervals,True

#Sample neighbor
def break_interval(point_break,intervals,block,nval,tab_train,tab_val,step,key):
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),10)
    #Sample interval
    which_interval = get_interval(point_break[:,:-2],intervals)
    new_interval,break_interval = get_break_interval(point_break[:,:-2],intervals,which_interval,key[0,0])
    index_interval = jnp.where(which_interval)[0]
    b_break = block[index_interval]
    #Compute interval cover of complement
    where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
    where_fill = jax.random.permutation(jax.random.PRNGKey(key[1,0]), where_fill)
    cover_intervals = cover_break_interval(new_interval,where_fill)
    #Divide into two blocks
    division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0]-2,),replace = True))
    division_old = jax.random.choice(jax.random.PRNGKey(key[3,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (jnp.sum(block == b_break) - 1,),replace = True)
    #Update partition
    intervals,block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)
    #Compute error
    error = get_error_partition(tab_train,tab_val,intervals,block,nval,key[4,0])
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Unite two blocks
def unite_blocks(unite,intervals,block,nval,tab_train,tab_val,step,key):
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),10)
    #Get error
    which_intervals = jnp.where((block == unite[0]) + (block == unite[1]) > 0)[0]
    unite_intervals = intervals[which_intervals,:]
    intervals = jnp.delete(intervals,which_intervals,0)
    block = jnp.delete(block,which_intervals)
    #Try to reduce
    reduced = False
    while(not reduced):
        unite_intervals,reduced = reduce(unite_intervals)
    intervals = jnp.append(intervals,unite_intervals,0)
    block = jnp.append(block,jnp.repeat(jnp.min(unite),unite_intervals.shape[0]))
    block = block.at[jnp.where(block > jnp.max(unite))].set(block[jnp.where(block > jnp.max(unite))] - 1)
    #Compute error
    error = get_error_partition(tab_train,tab_val,intervals,block,nval,key[0,0])
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}

#Dismenber block
def dismenber_blocks(b_break,intervals,block,nval,tab_train,tab_val,step,key):
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),10)
    #Which intervals to dismenber
    max_block = jnp.max(block) + 1
    division_new = jnp.append(jnp.append(b_break,max_block),jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.append(b_break,max_block),shape = (jnp.sum(block == b_break)-2,),replace = True))
    division_new = jax.jax.random.permutation(jax.random.PRNGKey(key[1,0]),division_new)
    block = block.at[block == b_break].set(division_new)
    #Compute error
    error = get_error_partition(tab_train,tab_val,intervals,block,nval,key[2,0])
    if not step:
        return error
    else:
        return {'block': block,'intervals': intervals,'error': error}
