#Lattice descent on the Interval Parition Lattice
import jax
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import math
import time

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(train,val,test = None,sample = 10,key = 0):
#Start seed
key = jax.random.split(jax.random.PRNGKey(key),3*epochs)
k = 0

#Get frequency tables
d = train.shape[1] - 1
tab_train = dt.get_ftable(train)
tab_val = dt.get_ftable(val)
nval = jnp.sum(tab_val[:,-2:])
if test is not None:
    tab_test = dt.get_ftable(test)

#Gather frequency tables in one array
domain = jnp.unique(jnp.append(jnp.append(jnp.zeros((1,d + 1)),1 + jnp.zeros((1,d + 1)),0),jnp.append(train,val,0),0)[:,:-1],axis = 0)
index_train = jax.vmap(lambda x: jnp.where((domain == x).all(-1),jnp.array(list(range(domain.shape[0]))),0).sum())(tab_train[:,:-2])
index_val = jax.vmap(lambda x: jnp.where((domain == x).all(-1),jnp.array(list(range(domain.shape[0]))),0).sum())(tab_val[:,:-2])
domain = jnp.append(domain,jnp.zeros((domain.shape[0],4)),1)
domain = domain.at[index_train,-4].set(tab_train[:,-2])
domain = domain.at[index_train,-3].set(tab_train[:,-1])
domain = domain.at[index_val,-2].set(tab_val[:,-2])
domain = domain.at[index_val,-1].set(tab_val[:,-1])
domain = jnp.append(1 + jnp.zeros((domain.shape[0],1)),domain,1) #Set zero to limits of interval
domain = domain.at[0,0].set(0)
domain = domain.at[-1,0].set(0)

#Initial partition
intervals = -1 + jnp.zeros((1,d)) #Matrix with intervals
block = jnp.array([0]) #Vector with block of each interval
points = list() #List with the sample points in each interval
points.append(domain)
npoints_block = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each block
npoints_intervals = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each interval
intervals_error = jnp.array(ut.error_block_partition(points[0],nval,key[k,0])).reshape((1,)) #Validation error of each block
k = k + 1

#Store error
current_error = intervals_error
best_error = current_error.copy()
best_intervals = intervals.copy()
best_block = block.copy()

#For each epoch
for e in range(epochs):
#Sampling probabilities for greater or smaller neighbors
small = jnp.array(math.comb(npoints.shape[0],2))
great = jnp.sum(npoints)
p1 = jnp.append(small,great)

#Sample neighbors
for n in range(s):
#Small or greater
if jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array([True,False]),shape=(1,),p = p1/jnp.sum(p1)):
    #Smaller neighbor
    k = k + 1
    #Which to unite
    unite = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block)))),shape=(2,),replace = False)
    k = k + 1
    #To be continued...
else:
#Which block to break
b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(np.max(block) + 1))),shape=(1,),p = npoints_block/jnp.sum(npoints_block))
k = k + 1
#Break a block
for k in range(100):
    print(k)
    b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = npoints_block/jnp.sum(npoints_block))
    tinit = time.time()
    block,intervals,points,npoints_block,npoints_intervals = ut.take_step(b_break,intervals,block,points.copy(),npoints_block,npoints_intervals,nval,key[k,0])
    print(time.time() - tinit)
    #err = ut.evaluate_neighbor(b_break,intervals,block,points.copy(),npoints_block,npoints_intervals,intervals_error,nval,key[k,0])
    #print(time.time() - tinit)

    #print(jnp.sum(intervals_error))
    if jnp.max(block) != k + 1 or jnp.min(block) != 0:
        break
    if intervals.shape[0] == 1:
        break
    if jnp.min(jnp.sum(intervals == -1,1)) == 0:
        break
    if jnp.unique(jnp.vstack(points),axis = 0).shape[0] != domain.shape[0]:
        break
    #if jnp.sum(npoints_block) != domain.shape[0]:
    #    break
    #if jnp.sum(npoints_intervals) != domain.shape[0]:
    #    break
    if len(block) != intervals.shape[0]:
        break
    if len(npoints_intervals) != intervals.shape[0]:
        break
    if len(npoints_block) != jnp.max(block) + 1:
        break

key = key[k,0]
#Error
nei_error = current_error.copy()
ut.error_block_partition(points[0],nval,key[k,0])
#
