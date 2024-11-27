#Lattice descent on the Interval Parition Lattice
import jax
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import math
import time
from alive_progress import alive_bar

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(epochs,train,val,batches = 1,batch_val = False,test = None,sample = 10,key = 0,unique = False,test_phase = False):
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
        tab_test = dt.get_ftable(test,unique)

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

    #Batches Size
    bsize = math.floor(domain.shape[0]/batches)

    #Initial partition
    print('- Initializing objects')
    intervals = -1 + jnp.zeros((1,d)) #Matrix with intervals
    block = jnp.array([0]) #Vector with block of each interval

    #Store error
    current_error = ut.get_error_partition(domain,intervals,block,nval,key[k,0])
    k = k + 1
    best_error = current_error
    best_intervals = intervals.copy()
    best_block = block.copy()

    #For each epoch
    print('- Starting epochs')
    tinit = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            print(' Error: ' + str(round(best_error,3)))
            domain = jax.random.permutation(jax.random.PRNGKey(key[k,0]), domain,0)
            k = k + 1
            for b in range(batches):
                if b < batches - 1:
                    tab_batch = domain[((b-1)*bsize):(b*bsize),:]
                else:
                    tab_batch = domain[((b-1)*bsize):,:]
                bnval = jnp.sum(tab_batch[:,-2:])
                #Sample neighbors
                error_batch = []
                kn = []
                move = []
                #Compute probabilities
                #TDB
                for n in range(sample):
                    #Sampling probabilities for greater or smaller neighbors
                    small = jnp.array(math.comb(jnp.max(block) + 1,2))
                    great = 100#jnp.sum(npoints_block)
                    p1 = jnp.append(small,great)
                    #Small or greater
                    if jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array([True,False]),shape=(1,),p = p1/jnp.sum(p1)):
                        #Smaller neighbor
                        k = k + 1
                        #Which to unite
                        unite = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                        k = k + 1
                        error_nei = ut.unite_blocks(unite,intervals,block,bnval,tab_batch,step = False,key = key[k,0])
                        k = k + 1
                        error_batch = error_batch + [error_nei.tolist()]
                        kn = kn + [k - 2]
                        move = move + ['unite']
                    else:
                        k = k + 1
                        #Which block to break
                        b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,))#,p = npoints_block/jnp.sum(npoints_block))
                        k = k + 1
                        error_nei = ut.sample_neighbor(b_break,intervals,block,bnval,tab_batch,step = False,key = key[k,0])
                        k = k + 1
                        error_batch = error_batch + [error_nei.tolist()]
                        kn = kn + [k - 2]
                        move = move + ['break']
                #Update partition
                error_batch = jnp.array(error_batch)
                kn = jnp.array(kn)
                kn = kn[error_batch == jnp.min(error_batch)][0]
                move = move[jnp.where(error_batch == jnp.min(error_batch))[0][0]]
                if move == 'break':
                    b_break = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,))#,p = npoints_block/jnp.sum(npoints_block))
                    kn = kn + 1
                    block,intervals = ut.sample_neighbor(b_break,intervals,block,bnval,tab_batch,step = True,key = key[kn,0])
                else:
                    unite = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                    kn = kn + 1
                    block,intervals = ut.unite_blocks(unite,intervals,block,bnval,tab_batch,step = True,key = key[kn,0])
            #Get error current partition
            current_error = ut.get_error_partition(domain,intervals,block,nval,key[k,0])
            k = k + 1
            #Store as best
            if current_error < best_error:
                best_error = current_error
                best_intervals = intervals.copy()
                best_block = block.copy()
            if test_phase:
                if jnp.min(jnp.sum(intervals == -1,1)) == 0:
                    print('E2')
                    break
                if len(block) != intervals.shape[0]:
                    print('E4')
                    break
            bar()
    return block,intervals,best_error
