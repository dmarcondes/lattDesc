#Utility functions lattDesc
import jax
from jax import numpy as jnp
import numpy as np
import time
from itertools import compress

#Test if element in interval
@jax.jit
def test_interval(interval,x):
    fixed = jnp.where(interval >= 0,1,0)
    return jnp.sum(jnp.where(fixed == 1,x != interval,False)) == 0

#Test if element not in interval
jax.jit
def test_not_interval(interval,x):
    fixed = jnp.where(interval >= 0,1,0)
    return jnp.sum(jnp.where(fixed == 1,x != interval,False)) != 0

#Teste if element is limit of interval (we know it is in interval)
@jax.jit
def test_limit_interval(interval,x):
    x_max = jnp.where(interval < 0,x,-1)
    x_min = jnp.where(interval < 0,x,1)
    return jnp.max(x_max) == jnp.min(x_min)

#Get elements in interval
@jax.jit
def get_elements_interval(interval,data):
    return jax.vmap(lambda x: test_interval(interval,x))(data)

#Get elements in some interval
@jax.jit
def get_elements_some_interval(intervals,data):
    return jnp.sum(jax.vmap(lambda interval: get_elements_interval(interval,data))(intervals),0) > 0

#Compute error of a block
@jax.jit
def error_block_partition(tab,nval,key):
    #Estimate class
    freq = jnp.sum(tab[:,-4:-2],0)
    pred = jnp.where(freq == jnp.max(freq),False,True)
    pred = pred.at[0].set(jax.random.choice(jax.random.PRNGKey(key), jnp.array([False,True]),shape=(1,),p = 1 - pred)[0])
    pred = pred.at[1].set(jax.random.choice(jax.random.PRNGKey(key+1), jnp.array([False,True]),shape=(1,),p = jnp.append(pred[0],1 - pred[0]))[0])
    freq_val = jnp.sum(tab[:,-2:],0)
    err = jnp.where(pred,freq_val,0)
    return jnp.sum(err)/nval

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
    return new_interval

#Get interval as inf
@jax.jit
def sample_inf(new_interval,break_interval):
    return jnp.where((break_interval == -1.0)*(new_interval == 0.0),-1.0,new_interval)

#Sample interval
def sample_interval(b_break,intervals,block,npoints_intervals,points,key):
    #Sample an interval in this block to break on
    index_interval = jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.array(list(range(intervals.shape[0]))),shape=(1,),p = jnp.where(block == b_break,npoints_intervals,0))
    break_interval = intervals[index_interval,:]
    #Sample a point in this interval to break on
    point = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array(list(range(points[index_interval[0]].shape[0]))),shape=(1,),p = points[index_interval[0]][:,0])
    #Break inf or sup
    new_interval = points[index_interval[0]][point,1:-4]
    inf_sup = jnp.repeat(jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape=(1,)),new_interval.shape[1])
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
    return intervals,block,max_block

#One step reduction
@jax.jit
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
def sample_neighbor(b_break,intervals,block,points,npoints_block,npoints_intervals,block_error,nval,domain,step,key):
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),100000)

    #Sample interval
    new_interval,break_interval,index_interval = sample_interval(b_break,intervals,block,npoints_intervals,points,key)

    #Compute interval cover of complement
    where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
    where_fill = jax.random.permutation(jax.random.PRNGKey(key[3,0]), where_fill)
    wf_size = where_fill.shape[0]
    where_fill = jnp.append(jnp.zeros((new_interval.shape[1] - wf_size)),where_fill).astype(jnp.int32)
    cover_intervals = cover_break_interval(new_interval,where_fill)[-(wf_size + 1):,:]

    #Divide into two blocks
    division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[4,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0]-2,),replace = True))
    division_old = jax.random.choice(jax.random.PRNGKey(key[5,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (jnp.sum(block == b_break)-1,),replace = True)

    #Update partition
    intervals,block,max_block = update_partition(b_break,intervals,cover_intervals,block,index_interval,division_old,division_new)

    #Compute error
    part1 = get_elements_some_interval(intervals[block == b_break,],domain[:,1:-4])
    e1 = error_block_partition(domain[part1,:],nval,key[6,0])
    part2 = get_elements_some_interval(intervals[block == jnp.max(block),],domain[:,1:-4])
    e2 = error_block_partition(domain[part2,:],nval,key[7,0])
    block_error = block_error.at[b_break].set(e1)
    block_error = jnp.append(block_error,e2)

    if not step:
        return jnp.sum(block_error)
    else:
        #Update points
        old_points = points[index_interval[0]].copy()
        del points[index_interval[0]]
        npoints_intervals = jnp.delete(npoints_intervals,index_interval[0])
        print('Entrou')
        tinit = time.time()
        for i in range(cover_intervals.shape[0]):
            tmp_domain = old_points[get_elements_interval(cover_intervals[i,:],old_points[:,1:-4]),:]
            tmp_domain = tmp_domain.at[:,0].set(1.0)
            tmp_domain = tmp_domain.at[jax.vmap(lambda x: test_limit_interval(cover_intervals[i,:],x))(tmp_domain[:,1:-4]),0].set(0)
            points.append(tmp_domain)
            npoints_intervals = jnp.append(npoints_intervals,jnp.sum(points[-1][:,0]))

        print(time.time() - tinit)
        npoints_block = npoints_block.at[b_break].set(jnp.sum(npoints_intervals[block == b_break]))
        npoints_block = jnp.append(npoints_block,jnp.sum(npoints_intervals[block == jnp.max(block)]))
    return block,intervals,points,npoints_block,npoints_intervals,block_error

#Unite two blocks
def unite_blocks(unite,intervals,block,points,npoints_block,npoints_intervals,block_error,nval,domain,step,key):
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),100)
    #Get error
    which_intervals = jnp.where((block == unite[0]) + (block == unite[1]) > 0)[0]
    new_points = jnp.vstack(list(compress(points,(block == unite[0]) + (block == unite[1]) > 0)))
    new_error = error_block_partition(new_points,nval,key[0,0])
    if not step:
        return jnp.sum(jnp.delete(block_error,unite)) + new_error
    else:
        #Get points in intervals
        for i in jnp.flip(which_intervals):
            del points[i]
        #Erase
        unite_intervals = intervals[which_intervals,:]
        intervals = jnp.delete(intervals,which_intervals,0)
        block = jnp.delete(block,which_intervals)
        npoints_intervals = jnp.delete(npoints_intervals,which_intervals)
        #Try to reduce
        reduced = False
        while(not reduced):
            unite_intervals,reduced = reduce(unite_intervals)
        intervals = jnp.append(intervals,unite_intervals,0)
        block = jnp.append(block,jnp.repeat(jnp.min(unite),unite_intervals.shape[0]))
        block = block.at[jnp.where(block > jnp.max(unite))].set(block[jnp.where(block > jnp.max(unite))] - 1)
        #Add points
        for i in range(unite_intervals.shape[0]):
            tmp_domain = new_points[get_elements_interval(unite_intervals[i,:],new_points[:,1:-4]),:]
            tmp_domain = tmp_domain.at[:,0].set(1.0)
            tmp_domain = tmp_domain.at[jax.vmap(lambda x: test_limit_interval(unite_intervals[i,:],x))(tmp_domain[:,1:-4]),0].set(0)
            points.append(tmp_domain)
            npoints_intervals = jnp.append(npoints_intervals,jnp.sum(points[-1][:,0]))
        block_error =  block_error.at[jnp.min(unite)].set(new_error)
        block_error = jnp.delete(block_error,jnp.max(unite))
        npoints_block = npoints_block.at[jnp.min(unite)].set(jnp.sum(npoints_intervals[block == jnp.min(unite)]))
        npoints_block = jnp.delete(npoints_block,jnp.max(unite))
        return block,intervals,points,npoints_block,npoints_intervals,block_error
