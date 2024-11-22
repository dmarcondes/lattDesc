#Utility functions lattDesc
import jax
from jax import numpy as jnp
from itertools import product

#Teste if element in interval
def test_interval(interval,x):
    fixed = interval >= 0
    if jnp.sum(fixed) == 0:
        return True
    else:
        return jnp.sum(x[fixed] != interval[fixed]) == 0

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

#Break a block
def break_block(b_break,intervals,block,points,number_points_block,number_points_intervals,,key):
#Seed
key = jax.random.split(jax.random.PRNGKey(key),3)

#Sample an interval in this block to break on
which_interval = jax.random.choice(jax.random.PRNGKey(key[0,0]), jnp.array(list(range(jnp.sum(block == b_break)))),shape=(1,))
index_interval = jnp.where(block == b_break)[0] + which_interval
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
what_fill = jnp.array([i for i in product(range(2),repeat = where_fill.shape[0])])
what_fill = what_fill[(what_fill != new_interval[:,where_fill]).any(-1),:]

interval_cover = jax.vmap(lambda x: intervals[block_break].at[:,where_fill].set(x))(what_fill).reshape((what_fill.shape[0],new_interval.shape[1]))
interval_cover = jax.random.permutation(jax.random.PRNGKey(key[2,0]), interval_cover)
reduced = False
while(not reduced):
    breaker = False
    for i in range(interval_cover.shape[0] - 1):
        if breaker:
            break
        for j in range(i + 1,interval_cover.shape[0]):
            if (interval_cover[i,:] <= interval_cover[j,:]).all() and jnp.sum(interval_cover[i,:] < interval_cover[j,:]) == 1:
                add = interval_cover[j,:]
                add = add.at[jnp.where((interval_cover[i,:] < interval_cover[j,:]))].set(-1.0).reshape((1,interval_cover.shape[1]))
                interval_cover = jnp.append(interval_cover,add,0)
                interval_cover = jnp.delete(interval_cover,jnp.array([i,j]),axis = 0)
                breaker = True
                break
            elif (interval_cover[j,:] <= interval_cover[i,:]).all() and jnp.sum(interval_cover[j,:] < interval_cover[i,:]) == 1:
                add = interval_cover[i,:]
                add = add.at[jnp.where((interval_cover[j,:] < interval_cover[i,:]))].set(-1.0).reshape((1,interval_cover.shape[1]))
                interval_cover = jnp.append(interval_cover,add,0)
                interval_cover = jnp.delete(interval_cover,jnp.array([i,j]),axis = 0)
                breaker = True
                break
    if j == interval_cover.shape[0] - 1 and i == interval_cover.shape[0] - 2 or interval_cover.shape[0] < 2:
        reduced = True

#Divide into two blocks
interval_cover = jnp.append(new_interval,interval_cover,0)
interval_cover = jnp.append(interval_cover,jnp.append(jnp.append(jnp.array(block_break),jnp.max(block) + 1).reshape((2,1)),jax.random.choice(jax.random.PRNGKey(key[3,0]),jnp.append(jnp.array(block_break),jnp.max(block) + 1),shape = (interval_cover.shape[0] - 2,1)),0),1)

#Update partition
intervals = jnp.append(intervals,interval_cover[:,:-1],0)
intervals = jnp.delete(intervals,block_break,axis = 0)
block = jnp.append(block,interval_cover[:,-1])
block = jnp.delete(block,block_break)
interval = interval_cover[:,0]
points_part_domain = points_part_domain.at[jnp.where(points_part_domain[:,-1] == block_break),-1].set(-1.0)
points_part_train = points_part_train.at[jnp.where(points_part_train[:,-1] == block_break),-1].set(-1.0)
points_part_val = points_part_val.at[jnp.where(points_part_val[:,-1] == block_break),-1].set(-1.0)
for i in range(interval_cover.shape[0]):
    points_in_train = jax.vmap(lambda x: test_interval(interval_cover[i,:-1],x))(points_part_train[:,:-1])
    if jnp.sum(points_in_train) > 0:
        points_part_train = points_part_train.at[jnp.where(points_in_train),-1].set(interval_cover[i,-1])
    points_in_val = jax.vmap(lambda x: test_interval(interval_cover[i,:-1],x))(points_part_val[:,:-1])
    if jnp.sum(points_in_val) > 0:
        points_part_val = points_part_val.at[jnp.where(points_in_val),-1].set(interval_cover[i,-1])
    points_in_domain = jax.vmap(lambda x: test_interval(interval_cover[i,:-1],x))(points_part_domain[:,:-1])
    if jnp.sum(points_in_domain) > 0:
        points_part_domain = points_part_domain.at[jnp.where(points_in_domain),-1].set(interval_cover[i,-1])

number_points = jnp.append(number_points.at[block_break].set(jnp.sum(points_part_domain[:,-1] == block_break)),jnp.sum(points_part_domain[:,-1] == jnp.max(block)))
return {'block': block,'intervals': intervals,'points_part_train': points_part_train,'points_part_val': points_part_val,'points_part_domain': points_part_domain,'number_points': number_points}
