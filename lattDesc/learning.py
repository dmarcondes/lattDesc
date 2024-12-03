#Lattice descent on the Interval Parition Lattice
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import math
import time
from alive_progress import alive_bar

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(train,val,epochs = 10,sample = 10,batches = 1,batch_val = False,test = None,num_classes = 2,key = 0,unique = False,intervals = None,block = None):
    """
    Stochastic Lattice Descent Algorithm in the Boolean Interval Partition Lattice
    -------
    Parameters
    ----------
    train : jax.numpy.array

        Array with training data. The last column contains the labels

    val : jax.numpy.array

        Array with validation data. The last column contains the labels

    epochs : int

        Training epochs

    sample : int

        Number of neighbors to sample at each step

    batches : int

        Number of sample batches in each epoch

    batch_val : logical

        Whether to consider batches for the validation data

    test : jax.numpy.array

        Array with test data. The last column contains the labels

    num_classes : int

        Number of classes

    key : int

        Seed for sampling

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    intervals : jax.numpy.array

        Array of initial intervals. If None then initialize with the unitary partition

    block : jax.numpy.array

        Array with the blocks of the initial intervals. If None then initialize with the unitary partition

    Returns
    -------
    dictionary with the learned 'block','intervals','best_error' and 'test_error', and the trace of the error ('trace_error') and time ('trace_time') over the epochs
    """
    print('------Starting algorithm------')
    #Start seed
    key = jax.random.split(jax.random.PRNGKey(key),10*epochs)
    k = 0

    #Get frequency tables
    print('- Creating frequency tables')
    d = train.shape[1] - 1 #dimension
    tab_train = dt.get_ftable(train,unique,num_classes) #Training table
    tab_val = dt.get_ftable(val,unique,num_classes) #Validation table
    nval = val.shape[0] #Validation sample size
    if test is not None: #Get test table if there is test data
        tab_test = dt.get_ftable(test,unique,num_classes)

    #Batches Size
    if unique: #If data is unique, batches of frequency table
        bsize = math.floor(tab_train.shape[0]/batches) #Batch size for training
        bsize_val = math.floor(tab_val.shape[0]/batches) #Batch size for validation
    else: #Else, batches of data
        bsize = math.floor(train.shape[0]/batches) #Batch size for training
        bsize_val = math.floor(val.shape[0]/batches) #Batch size for validation

    #Initial partition
    print('- Initializing objects')
    if intervals is None or block is None: #If initial partition is not given, initialize
        intervals = -1 + jnp.zeros((1,d)) #Array with intervals
        block = jnp.array([0]) #Array with block of each interval

    #Store error
    current_error = ut.get_error_partition(tab_train,tab_val,intervals,block,nval,key[k,0],num_classes) #Get error
    k = k + 1 #Update seed
    best_error = current_error #Best error_batch
    best_intervals = intervals.copy() #Best intervals
    best_block = block.copy() #Best block

    #Objects to trace
    trace_error = jnp.array([]) #Trace algorithm time
    trace_time = jnp.array([]) #Trace algorithm error

    #For each epoch
    print('- Starting epochs')
    tinit = time.time() #Initialize time
    print(' Initial error: ' + str(round(best_error,3))) #Prits initial error
    with alive_bar(epochs) as bar: #Alive bar for tracing
        for e in range(epochs): #For each epoch
            if batches > 1: #If there should be training batches
                if unique: #If data is unique, batches of frequency table
                    tab_train = jax.random.permutation(jax.random.PRNGKey(key[k,0]), tab_train,0) #Random permutation of training table
                    k = k + 1 #Update seed
                    tab_val = jax.random.permutation(jax.random.PRNGKey(key[k,0]), tab_val,0) #Random permutation of validation table
                    k = k + 1 #Update seed
                else: #Batches of data
                    train = jax.random.permutation(jax.random.PRNGKey(key[k,0]), train,0) #Random permutation of training data
                    k = k + 1 #Update seed
                    val = jax.random.permutation(jax.random.PRNGKey(key[k,0]), val,0) #Random permutation of validation data
                    k = k + 1 #Update seed
            for b in range(batches): #For each batch
                #Get frequency table of batch
                tab_train_batch,tab_val_batch,bnval = ut.get_tfrequency_batch(b,batches,tab_train,tab_val,train,val,bsize,bsize_val,unique,batch_val,nval,num_classes)

                #Compute probabilities
                small = jnp.array(math.comb(jnp.max(block) + 1,2)) #Number of ways of uniting intervals
                dismenber = jnp.power(jnp.bincount(block) - 1,2) - 1 #Number of ways of dimenbering
                breakInt = ut.get_limits_some_interval(intervals,tab_train_batch[:,0:-num_classes]) #Flag intervals that are limit of intervals
                prob = jnp.append(jnp.append(small,jnp.sum(dismenber)),jnp.sum(1 - breakInt)) #Probability of uniting, diemenbering and breaking interval al internal point
                what_nei = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array([0,1,2]),shape=(sample,),p = prob) #Sample kind of step to take at each sample neighbor
                k = k + 1 #Update seed

                #Objects to store neighbors
                store_nei = list() #Store neighbors
                error_batch = jnp.array([]) #Store error

                #Sample neighbors
                for n in range(sample): #For each neighbor
                    if what_nei[n] == 0: #Unite intervals
                        #Sample block to unite intervals
                        unite = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(2,),replace = False)
                        k = k + 1 #Update seed

                        #Sample intervals to unite in the sampled block and store the result
                        store_nei.append(ut.unite_blocks(unite,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = jnp.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 1: #Dismenber intervals
                        #Sample block to dismenber intervals
                        b_dis = jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(jnp.max(block) + 1))),shape=(1,),p = dismenber)
                        k = k + 1 #Update seed

                        #Sample dismenbering of the sampled block and store the result
                        store_nei.append(ut.dismenber_blocks(b_dis,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = jnp.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 2: #Break interval
                        #Sample point to break an interval of the sampled block on
                        point_break = tab_train_batch[jax.random.choice(jax.random.PRNGKey(key[k,0]), jnp.array(list(range(tab_train_batch.shape[0]))),shape=(1,),p = 1 - breakInt),:]
                        k = k + 1 #Update seed

                        #Break interval on sampled point and store the result
                        store_nei.append(ut.break_interval(point_break,intervals,block,bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = jnp.append(error_batch,store_nei[-1]['error'])
                #Update partition at each batch
                which_nei = jnp.where(error_batch == jnp.min(error_batch))[0][0] #Get first neighbor with the least error
                block = store_nei[which_nei]['block'] #Update block
                intervals = store_nei[which_nei]['intervals'] #Update interval
                del store_nei, error_batch #Delete trace of neighbors
            #Get error current partition at the end of epoch
            current_error = ut.get_error_partition(tab_train,tab_val,intervals,block,nval,key[k,0],num_classes)
            k = k + 1 #Update seed

            #Store current partition as best with it has the least error so far
            if current_error < best_error:
                best_error = current_error #Store error
                best_intervals = intervals.copy() #Store intervals
                best_block = block.copy() #Store blocks
                print('Error: ' + str(round(best_error,3))) #Print error

            #Trace
            trace_error = jnp.append(trace_error,current_error) #Trace error
            trace_time = jnp.append(trace_time,jnp.array([time.time() - tinit])) #Trace time
            bar() #Update bar
    #Test error
    test_error = None #Initialize test error
    if test is not None: #Compute test error if there is test data
        test_error = ut.get_error_partition(tab_train,tab_test,intervals,block,test.shape[0],key[k,0],num_classes)

    #Return
    return {'block': block,'intervals': intervals,'best_error': best_error,'test_error': test_error,'trace_error': trace_error,'trace_time': trace_time}
