#Utility functions lattDesc
import jax
from jax import numpy as jnp
import numpy as np
import time

#Teste if element in interval
def test_interval(interval,x):
    fixed = interval >= 0
    if jnp.sum(fixed) == 0:
        return True
    else:
        return jnp.sum(x[fixed] != interval[fixed]) == 0

#Teste if element is limit of interval (we know it is in interval)
def test_limit_interval(interval,x):
    free = interval < 0
    return jnp.max(x[free]) == jnp.min(x[free])

#Get elements in interval
def get_elements_interval(interval,data):
    return data[jax.vmap(lambda x: test_interval(interval,x))(data),:]

#Compute error of a block
def error_block_partition(tab,nval,key):
    #Estimate class
    freq = jnp.sum(tab[:,-4:-2],0)
    if freq[0] > freq[1]:
        pred = 0
    elif freq[0] < freq[1]:
        pred = 1
    else:
        pred = jax.random.choice(jax.random.PRNGKey(key), jnp.array([0,1]),shape=(1,))
    #Compute validation error
    freq_val = jnp.sum(tab[:,-2:],0)
    if pred == 0:
        error_block = freq_val[1]/nval
    else:
        error_block = freq_val[0]/nval
    return error_block

#Compute error of patition
def error_partition(points_part_train,points_part_val,block,tab_train,tab_val,key):
    #Estimate class of each block
    freq = jax.vmap(lambda b: get_sum_frequences_block(b,tab_train,points_part_train))(block)
    pred = jax.vmap(lambda f: jnp.where(f == jnp.max(f),jnp.array([1,2]),0).sum())(freq)
    if jnp.sum(pred == 3) > 0:
        pred = pred.at[jnp.where(pred == 3)].set(jax.random.choice(jax.random.PRNGKey(key), jnp.array([1,2]),shape=(jnp.sum(pred == 3),)))
    pred = pred.at[jnp.where(pred == 1)].set(0)
    pred = pred.at[jnp.where(pred == 2)].set(1)
    #Compute validation error
    freq_val = jax.vmap(lambda b: get_sum_frequences_block(b,tab_val,points_part_val))(block)
    error_blocks = jax.vmap(lambda i,f: f[i])(1 - pred,freq_val)/jnp.sum(tab_val[:,-2:])
    return jnp.sum(error_blocks)

#Take a step
def take_step(b_break,intervals,block,points,npoints_block,npoints_intervals,nval,key):
    tinit = time.time()
    #Seed
    key = jax.random.split(jax.random.PRNGKey(key),100000)

    #Sample an interval in this block to break on
    which_interval = jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.array(list(range(jnp.sum(block == b_break)))),shape=(1,),p = npoints_intervals[block == b_break])
    index_interval = jnp.where(block == b_break)[0][which_interval]
    break_interval = intervals[index_interval,:]

    #Sample a point in this interval to break on
    point = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array(list(range(points[index_interval[0]].shape[0]))),shape=(1,),p = points[index_interval[0]][:,0])

    #Break inf or sup
    inf_sup = jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape = (1,))
    new_interval = points[index_interval[0]][point,1:-4]
    if inf_sup == 1: #Sup limit of interval
        new_interval = new_interval.at[jnp.where((break_interval == -1.0)*(new_interval == 1.0))].set(-1.0)
    else: #Inf limit of interval
        new_interval = new_interval.at[jnp.where((break_interval == -1.0)*(new_interval == 0.0))].set(-1.0)
    #Compute interval cover of complement
    where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
    cover_intervals = None
    for i in range(len(where_fill)):
        tmp = new_interval
        tmp = tmp.at[0,where_fill[i]].set(1 - tmp[0,where_fill[i]])
        tmp = tmp.at[0,where_fill[(i+1):]].set(-1)
        if cover_intervals is None:
            cover_intervals = tmp
        else:
            cover_intervals = jnp.append(cover_intervals,tmp,0)
    cover_intervals = jnp.append(new_interval,cover_intervals,0)

    #Divide into two blocks
    division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[3,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0]-2,),replace = True))
    division_old = jax.random.choice(jax.random.PRNGKey(key[4,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (jnp.sum(block == b_break)-1,),replace = True)

    #Update partition
    intervals = jnp.append(intervals,cover_intervals,0)
    intervals = jnp.delete(intervals,index_interval,0)
    max_block = jnp.max(block)
    block = jnp.delete(block,index_interval)
    block = block.at[jnp.where(block == b_break)].set(division_old)
    block = jnp.append(block,jnp.where(division_new == 0,b_break,max_block + 1))

    old_points = points[index_interval[0]].copy()
    del points[index_interval[0]]
    npoints_intervals = jnp.delete(npoints_intervals,index_interval[0])
    for i in range(cover_intervals.shape[0]):
        tmp_domain = old_points[jax.vmap(lambda x: test_interval(cover_intervals[i,:],x))(old_points[:,1:-4]),:]
        tmp_domain = tmp_domain.at[:,0].set(1.0)
        tmp_domain = tmp_domain.at[jax.vmap(lambda x: test_limit_interval(cover_intervals[i,:],x))(tmp_domain[:,1:-4]),0].set(0)
        points.append(tmp_domain)
        npoints_intervals = jnp.append(npoints_intervals,jnp.sum(points[-1][:,0]))

    npoints_block = npoints_block.at[b_break].set(jnp.sum(npoints_intervals[block == b_break]))
    npoints_block = jnp.append(npoints_block,jnp.sum(npoints_intervals[block == jnp.max(block)]))
    return block,intervals,points,npoints_block,npoints_intervals

def evaluate_neighbor(b_break,intervals,block,points,npoints_block,npoints_intervals,intervals_error,nval,key):
#Seed
key = jax.random.split(jax.random.PRNGKey(key),100000)

#Sample an interval in this block to break on
which_interval = jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.array(list(range(jnp.sum(block == b_break)))),shape=(1,),p = npoints_intervals[block == b_break])
index_interval = jnp.where(block == b_break)[0][which_interval]
break_interval = intervals[index_interval,:]

#Sample a point in this interval to break on
point = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array(list(range(points[index_interval[0]].shape[0]))),shape=(1,),p = points[index_interval[0]][:,0])

#Break inf or sup
inf_sup = jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array([0,1]),shape = (1,))
new_interval = points[index_interval[0]][point,1:-4]
if inf_sup == 1: #Sup limit of interval
    new_interval = new_interval.at[jnp.where((break_interval == -1.0)*(new_interval == 1.0))].set(-1.0)
else: #Inf limit of interval
    new_interval = new_interval.at[jnp.where((break_interval == -1.0)*(new_interval == 0.0))].set(-1.0)

#Compute interval cover of complement
where_fill = jnp.where((new_interval != -1.0)*(break_interval == -1.0))[1]
cover_intervals = None
for i in range(len(where_fill)):
    tmp = new_interval
    tmp = tmp.at[0,where_fill[i]].set(1 - tmp[0,where_fill[i]])
    tmp = tmp.at[0,where_fill[(i+1):]].set(-1)
    if cover_intervals is None:
        cover_intervals = tmp
    else:
        cover_intervals = jnp.append(cover_intervals,tmp,0)

cover_intervals = jnp.append(new_interval,cover_intervals,0)

#Divide into two blocks
division_new = jnp.append(jnp.array([1,0]),jax.random.choice(jax.random.PRNGKey(key[3,0]), jnp.array([0,1]),shape = (cover_intervals.shape[0]-2,),replace = True))
division_old = jax.random.choice(jax.random.PRNGKey(key[4,0]), jnp.append(b_break,jnp.max(block) + 1),shape = (jnp.sum(block == b_break)-1,),replace = True)

#Compute error
intervals = jnp.append(intervals,cover_intervals,0)
intervals = jnp.delete(intervals,index_interval,0)
max_block = jnp.max(block)
block = jnp.delete(block,index_interval)
block = block.at[jnp.where(block == b_break)].set(division_old)
block = jnp.append(block,jnp.where(division_new == 0,b_break,max_block + 1))
old_points = points[index_interval[0]].copy()
del points[index_interval[0]]

error = jnp.array([])
for i in jnp.where((block == b_break) + (block == max_block) > 0):
    tmp_domain = old_points[jax.vmap(lambda x: test_interval(cover_intervals[i,:],x))(old_points[:,1:-4]),:]
    error = jnp.append(error,ut.error_block_partition(tmp_domain,nval,key[5+i,0]))
intervals_error.at[b_break].set(error)

return jnp.sum(intervals_error)
