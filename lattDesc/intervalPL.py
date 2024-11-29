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
    key = jax.random.split(jax.random.PRNGKey(key),10*epochs)
    k = 0

    #Get frequency tables
    print('- Creating frequency tables')
    d = train.shape[1] - 1
    tab_train = dt.get_ftable(train,unique)
    tab_val = dt.get_ftable(val,unique)
    nval = val.shape[0]
    if test is not None:
        tab_test = dt.get_ftable(test,unique)

    #Batches Size
    bsize = math.floor(tab_train.shape[0]/batches)
    bsize_val = math.floor(tab_val.shape[0]/batches)

    #Initial partition
    print('- Initializing objects')
    intervals = -1 + jnp.zeros((1,d)) #Matrix with intervals
    block = jnp.array([0]) #Vector with block of each interval

    #Store error
    current_error = ut.get_error_partition(tab_train,tab_val,intervals,block,nval,key[k,0])
    k = k + 1
    best_error = current_error
    best_intervals = intervals.copy()
    best_block = block.copy()

    #Objects to trace
    trace_error = []
    trace_time = []

    #For each epoch
    print('- Starting epochs')
    tinit = time.time()
    print(' Initial error: ' + str(round(best_error,3)))
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            tab_train = jax.random.permutation(jax.random.PRNGKey(key[k,0]), tab_train,0)
            k = k + 1
            tab_val = jax.random.permutation(jax.random.PRNGKey(key[k,0]), tab_val,0)
            k = k + 1
            for b in range(batches):
                if b < batches - 1:
                    tab_train_batch = tab_train[((b-1)*bsize):(b*bsize),:]
                else:
                    tab_train_batch = tab_train[((b-1)*bsize):,:]
                if batch_val:
                    if b < batches - 1:
                        tab_val_batch = tab_val[((b-1)*bsize):(b*bsize),:]
                    else:
                        tab_val_batch = tab_val[((b-1)*bsize):,:]
                    bnval = jnp.sum(tab_val_batch[:,-2:])
                else:
                    tab_val_batch = tab_val
                    bnval = nval
                #Sample neighbors
                error_batch = []
                kn = []
                move = []
                #Compute probabilities
                print('Compute probs')
                tinit = time.time()
                small = jnp.array(math.comb(jnp.max(block) + 1,2))
                dismenber = jnp.power(jnp.bincount(block) - 1,2) - 1
                breakInt = ut.get_limits_some_interval(intervals,tab_train[:,0:-2])
                what_nei = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array([0,1,2]),shape=(sample,),p = jnp.append(jnp.append(small,jnp.sum(dismenber)),jnp.sum(1 - breakInt)))
                k = k + 1
                print(time.time() - tinit)
                for n in range(sample):
                    #Unite
                    if what_nei[n] == 0:
                        print('Unite')
                        tinit = time.time()
                        #Which to unite
                        unite = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                        k = k + 1
                        #Compute error
                        error_nei = ut.unite_blocks(unite,intervals,block,bnval,tab_train_batch,tab_val_batch,step = False,key = key[k,0])
                        k = k + 1
                        #Save error
                        error_batch = error_batch + [error_nei.tolist()]
                        kn = kn + [k - 2]
                        move = move + ['unite']
                        print(time.time() - tinit)
                    elif what_nei[n] == 1:
                        print('Dismenber')
                        tinit = time.time()
                        #Which block to break
                        b_break = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = dismenber)
                        k = k + 1
                        #Compute error
                        error_nei = ut.dismenber_blocks(b_break,intervals,block,bnval,tab_train_batch,tab_val_batch,step = False,key = key[k,0])
                        k = k + 1
                        error_batch = error_batch + [error_nei.tolist()]
                        kn = kn + [k - 2]
                        move = move + ['dismenber']
                        print(time.time() - tinit)
                    elif what_nei[n] == 2:
                        print('Break')
                        #On which point to break
                        point_break = tab_train[jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(tab_train.shape[0]))),shape=(1,),p = 1 - breakInt),:]
                        k = k + 1
                        #Compute error
                        error_nei = ut.break_interval(point_break,intervals,block,bnval,tab_train_batch,tab_val_batch,step = False,key = key[k,0])
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
                    point_break = tab_train[jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(tab_train.shape[0]))),shape=(1,),p = 1 - breakInt),:]
                    kn = kn + 1
                    block,intervals = ut.break_interval(point_break,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[kn,0])
                elif move == 'unite':
                    unite = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                    kn = kn + 1
                    block,intervals = ut.unite_blocks(unite,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[kn,0])
                elif move == 'dismenber':
                    b_break = jax.random.choice(jax.random.PRNGKey(key[kn,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = dismenber)
                    kn = kn + 1
                    block,intervals  = ut.dismenber_blocks(b_break,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[kn,0])
            #Get error current partition
            current_error = ut.get_error_partition(tab_train,tab_val,intervals,block,nval,key[k,0])
            k = k + 1
            #Store as best
            if current_error < best_error:
                best_error = current_error
                best_intervals = intervals.copy()
                best_block = block.copy()
                print(' Error: ' + str(round(best_error,3)))
            if test_phase:
                if tab_train[ut.get_elements_some_interval(intervals,tab_train[:,0:-2,]),:].shape[0] != tab_train.shape[0]:
                    print('E1')
                    break
                if jnp.min(jnp.sum(intervals == -1,1)) == 0:
                    print('E2')
                    break
                if len(block) != intervals.shape[0]:
                    print('E4')
                    break
            trace_error = trace_error + [current_error.tolist()]
            trace_time = trace_time + [time.time() - tinit]
            bar()
    test_error = None
    if test is not None:
        test_error = ut.get_error_partition(tab_train,tab_test,intervals,block,test.shape[0],key[k,0])
    return block,intervals,best_error,test_error,trace_error,trace_time
