#Lattice descent on the Interval Parition Lattice
import jax
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import math
import time
from alive_progress import alive_bar

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(epochs,train,val,test = None,sample = 10,key = 0,unique = False):
    print('------Starting algorithm------')
    #Start seed
    key = jax.random.split(jax.random.PRNGKey(key),3*epochs)
    k = 0

    #Get frequency tables
    print('- Creating frequency tables')
    d = train.shape[1] - 1
    tab_train = dt.get_ftable(train,unique)
    tab_val = dt.get_ftable(val,unique)
    nval = val.shape[0]
    if test is not None:
        tab_test = dt.get_ftable(test)

    #Gather frequency tables in one array
    print('- Creating arrays')
    if not unique:
        domain = jnp.unique(jnp.append(jnp.append(jnp.zeros((1,d + 1)),1 + jnp.zeros((1,d + 1)),0),jnp.append(train,val,0),0)[:,:-1],axis = 0)
    else:
        domain = jnp.append(jnp.append(jnp.zeros((1,d + 1)),1 + jnp.zeros((1,d + 1)),0),jnp.append(train,val,0),0)[:,:-1]

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
    print('- Initializing objects')
    intervals = -1 + jnp.zeros((1,d)) #Matrix with intervals
    block = jnp.array([0]) #Vector with block of each interval
    points = list() #List with the sample points in each interval
    points.append(domain)
    npoints_block = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each block
    npoints_intervals = jnp.sum(domain[:,0]).reshape((1,)) #Vector with sample points in each interval
    block_error = jnp.array(ut.error_block_partition(points[0],nval,key[k,0])).reshape((1,)) #Validation error of each block
    k = k + 1

    #Store error
    current_error = jnp.sum(block_error)
    best_error = current_error
    best_intervals = intervals.copy()
    best_block = block.copy()

    #For each epoch
    print('- Starting epochs')
    tinit = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            print(' Error: ' + str(round(best_error,3)))
            #Sample neighbors
            error_epoch = []
            kn = []
            move = []
            for n in range(sample):
                #Sampling probabilities for greater or smaller neighbors
                small = jnp.array(math.comb(jnp.max(block) + 1,2))
                great = jnp.sum(npoints_block)
                p1 = jnp.append(small,great)
                #Small or greater
                if jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array([True,False]),shape=(1,),p = p1/jnp.sum(p1)):
                    #Smaller neighbor
                    k = k + 1
                    #Which to unite
                    unite = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                    k = k + 1
                    error_nei = ut.unite_blocks(unite,intervals,block,points.copy(),npoints_block,npoints_intervals,block_error,nval,domain,step = False,key = key[k,0])
                    k = k + 1
                    error_epoch = error_epoch + [error_nei.tolist()]
                    kn = kn + [k - 2]
                    move = move + ['unite']
                else:
                    k = k + 1
                    #Which block to break
                    b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = npoints_block/jnp.sum(npoints_block))
                    k = k + 1
                    error_nei = ut.sample_neighbor(b_break,intervals,block,points.copy(),npoints_block,npoints_intervals,block_error,nval,domain,step = False,key = key[k,0])
                    k = k + 1
                    error_epoch = error_epoch + [error_nei.tolist()]
                    kn = kn + [k - 2]
                    move = move + ['break']
            #Update partition
            error_epoch = jnp.array(error_epoch)
            current_error = jnp.min(error_epoch)
            kn = jnp.array(kn)
            kn = kn[error_epoch == jnp.min(error_epoch)][0]
            move = move[jnp.where(error_epoch == jnp.min(error_epoch))[0][0]]
            if move == 'break':
                b_break = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = npoints_block/jnp.sum(npoints_block))
                kn = kn + 1
                block,intervals,points,npoints_block,npoints_intervals,block_error = ut.sample_neighbor(b_break,intervals,block,points.copy(),npoints_block,npoints_intervals,block_error,nval,domain,step = True,key = key[kn,0])
            else:
                unite = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                kn = kn + 1
                block,intervals,points,npoints_block,npoints_intervals,block_error = ut.unite_blocks(unite,intervals,block,points.copy(),npoints_block,npoints_intervals,block_error,nval,domain,step = True,key = key[kn,0])
            #Store as best
            if current_error < best_error:
                best_error = current_error
                best_intervals = intervals.copy()
                best_block = block.copy()
            if jnp.abs(current_error - jnp.sum(block_error)) > 1e-06:
                print('E1')
                break
            if jnp.min(jnp.sum(intervals == -1,1)) == 0:
                print('E2')
                break
            if jnp.unique(jnp.vstack(points),axis = 0).shape[0] != domain.shape[0]:
                print('E3')
                break
            if len(block) != intervals.shape[0]:
                print('E4')
                break
            if len(npoints_intervals) != intervals.shape[0]:
                print('E5')
                break
            if len(npoints_block) != jnp.max(block) + 1:
                print('E6')
                break
            bar()
    return block,intervals,points,block_error
