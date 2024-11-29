#Utility functions lattDesc
import jax
from jax import numpy as jnp
import numpy as np
import time
from itertools import compress

#Test if element in interval
@jax.jit
def test_interval(interval,x):
    """
    Test if element is in interval
    -------

    Parameters
    ----------
    interval : jax numpy array

        Interval

    x : jax numpy array

        Element

    Returns
    -------

    logical

    """
    fixed = jnp.where(interval >= 0,1,0)
    return jnp.sum(jnp.where(fixed == 1,x != interval,False)) == 0

#Test if element not in interval
jax.jit
def test_not_interval(interval,x):
    """
    Test if element is not in interval
    -------

    Parameters
    ----------
    interval : jax numpy array

        Interval

    x : jax numpy array

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
    interval : jax numpy array

        Interval

    x : jax numpy array

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
    return jax.vmap(lambda x: test_limit_interval(interval,x))(data)

#Get elements that are limit of some interval
@jax.jit
def get_limits_some_interval(intervals,data):
    return jnp.sum(jax.vmap(lambda interval: get_limits_interval(interval,data))(intervals),0) > 0

#Get elements in interval
@jax.jit
def get_elements_interval(interval,data):
    """
    Flag the elements in dataset that are in a interval
    -------

    Parameters
    ----------
    interval : jax numpy array

        Interval

    data : jax numpy array

        Dataset

    Returns
    -------

    jax numpy array of logical

    """
    return jax.vmap(lambda x: test_interval(interval,x))(data)

#Get elements in some interval
@jax.jit
def get_elements_some_interval(intervals,data):
    """
    Flag the elements in dataset that are in some interval in a set
    -------

    Parameters
    ----------
    intervals : jax numpy array

        A set of intervals

    data : jax numpy array

        Dataset

    Returns
    -------

    jax numpy array of logical

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
    tab : jax numpy array

        Sample points that are in the block in the format (prob_sample,points,tab_train,tab_val)

    nval : int

        Size of validation sample

    key : int

        Key for random classification in the presence of ties

    Returns
    -------

    float

    """
    #Estimate class
    freq = jnp.sum(tab_train[:,-2:],0)
    pred = jnp.where(freq == jnp.max(freq),False,True)
    pred = pred.at[0].set(jax.random.choice(jax.random.PRNGKey(key), jnp.array([False,True]),shape=(1,),p = 1 - pred)[0])
    pred = pred.at[1].set(jax.random.choice(jax.random.PRNGKey(key+1), jnp.array([False,True]),shape=(1,),p = jnp.append(pred[0],1 - pred[0]))[0])
    freq_val = jnp.sum(tab_val[:,-2:],0)
    err = jnp.where(pred,freq_val,0)
    return jnp.sum(err)/nval

#Get error partition
@jax.jit
def get_error_partition(tab_train,tab_val,intervals,block,nval,key):
    error = 0
    for i in range(jnp.max(block) + 1):
        tab_train_block = tab_train[get_elements_some_interval(intervals[block == i,:],tab_train[:,0:-2,]),:]
        tab_val_block = tab_val[get_elements_some_interval(intervals[block == i,:],tab_val[:,0:-2,]),:]
        error = error + error_block_partition(tab_train_block,tab_val_block,nval,key)
    return error

#Break interval at new interval
@jax.jit
def cover_break_interval(new_interval,where_fill):
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
def sample_sup(new_interval,break_interval):
    return jnp.where((break_interval == -1.0)*(new_interval == 1.0),-1.0,new_interval)

#Get interval as inf
@jax.jit
def sample_inf(new_interval,break_interval):
    return jnp.where((break_interval == -1.0)*(new_interval == 0.0),-1.0,new_interval)

#Sample interval
def sample_interval(b_break,intervals,block,domain,key):
    #Sample an interval in this block to break on
    index_interval = jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.array(list(range(intervals.shape[0]))),shape=(1,),p = jnp.where(block == b_break,1,0))
    break_interval = intervals[index_interval,:]
    #Sample a point in this interval to break on
    points = domain[get_elements_interval(break_interval,domain[:,0:-2]),:]
    point = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array(list(range(points.shape[0]))),shape=(1,))#,p = points[index_interval[0]][:,0])
    #Break inf or sup
    new_interval = points[point,0:-2]
    inf_sup = jnp.repeat(jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape=(1,)),new_interval.shape[1])
    new_interval_sup = sample_inf(new_interval,break_interval)
    new_interval_inf = sample_sup(new_interval,break_interval)
    new_interval = jnp.where(inf_sup == 1,new_interval_sup,new_interval_inf)
    return new_interval,break_interval,index_interval

#Sample interval
def get_break_interval(point_break,intervals,key):
    #Sample an interval in this block to break on
    index_interval = jnp.where(jax.vmap(lambda interval: test_interval(interval,point_break[:,0:-2]))(intervals))[0]
    break_interval = intervals[index_interval,:]
    #Break inf or sup
    new_interval = point_break[:,0:-2]
    inf_sup = jnp.repeat(jax.random.choice(jax.random.PRNGKey(key), jnp.array([0,1]),shape=(1,)),new_interval.shape[1])
    new_interval_sup = sample_inf(new_interval,break_interval)
    new_interval_inf = sample_sup(new_interval,break_interval)
    new_interval = jnp.where(inf_sup == 1,new_interval_sup,new_interval_inf)
    return new_interval,break_interval,index_interval

#Update partition
def update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new):
    intervals = jnp.append(intervals,cover_intervals,0)
    intervals = jnp.delete(intervals,index_interval,0)
    max_block = jnp.max(block)
    block = jnp.delete(block,index_interval)
    where_old = jnp.where(block == b_break)
    block = block.at[where_old].set(division_old)
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
    print('Get break interval')
    tinit = time.time()
    new_interval,break_interval,index_interval = get_break_interval(point_break,intervals,key[0,0])
    b_break = block[index_interval]
    print(time.time() - tinit)
    #Compute interval cover of complement
    print('Compute interval cover complement')
    tinit = time.time()
    where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
    where_fill = jax.random.permutation(jax.random.PRNGKey(key[1,0]), where_fill)
    wf_size = where_fill.shape[0]
    where_fill = jnp.append(jnp.zeros((new_interval.shape[1] - wf_size)),where_fill).astype(jnp.int32)
    cover_intervals = cover_break_interval(new_interval,where_fill)[-(wf_size + 1):,:]
    print(time.time() - tinit)
    #Divide into two blocks
    print('Update partition')
    tinit = time.time()
    division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0]-2,),replace = True))
    division_old = jax.random.choice(jax.random.PRNGKey(key[3,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (jnp.sum(block == b_break)-1,),replace = True)
    #Update partition
    intervals,block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)
    print(time.time() - tinit)
    #Compute error
    print('Get error partition')
    tinit = time.time()
    error = get_error_partition(tab_train,tab_val,intervals,block,nval,key[4,0])
    print(time.time() - tinit)
    if not step:
        return error
    else:
        return block,intervals

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
        return block,intervals

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
        return block,intervals
