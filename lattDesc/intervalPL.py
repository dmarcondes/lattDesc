#Lattice descent on the Interval Parition Lattice
import jax
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import math
import numpy as np

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
block = np.array([0]) #Vector with block of each interval
points = list() #List with the sample points in each interval
points.append(domain)
number_points_block = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each block
number_points_intervals = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each interval
blocks_error = np.array(ut.error_block_partition(points[0],nval,key[k,0])).reshape((1,)) #Validation error of each block
k = k + 1

#Store error
current_error = blocks_error
best_error = current_error.copy()
best_intervals = intervals.copy()
best_block = block.copy()

#For each epoch
for e in range(epochs):
#Sampling probabilities for greater or smaller neighbors
small = jnp.array(math.comb(number_points.shape[0],2))
great = jnp.sum(number_points)
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
b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(np.max(block) + 1))),shape=(1,),p = number_points_block/jnp.sum(number_points_block))
k = k + 1
#Break a block
broken_block = ut.break_block(b_break,intervals,block,points,number_points_block,number_points_intervals,key[k,0])
k = k + 1

#Error
nei_error = current_error.copy()
nei_error = nei_error.at[jnp.where(block == block_break)].set(ut.error_block_partition(block_break,broken_block['points_part_train'],broken_block['points_part_val'],tab_train,tab_val,key[k,0]))
nei_error = jnp.append(nei_error,ut.error_block_partition(jnp.max(block) + 1,broken_block['points_part_train'],broken_block['points_part_val'],tab_train,tab_val,key[k,0]))

#
