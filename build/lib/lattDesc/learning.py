#Lattice descent on the Interval Parition Lattice
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import numpy as np
import math
import time
from alive_progress import alive_bar
import os

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(train,val,epochs = 10,sample = 10,batches = 1,batch_val = False,test = None,num_classes = 2,key = 0,unique = False,intervals = None,block = None,video = False,filename = 'video_sdesc_BIPL',framerate = 1):
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

    video : logical

        Wheter to generate a video with the algorithm steps

    filename : str

        File name for video

    framerate : int

        Framerate for video

    Returns
    -------
    dictionary with the learned 'block','intervals','best_error' and 'test_error', and the trace of the error ('trace_error') and time ('trace_time') over the epochs
    """
    print('------Starting algorithm------')
    #Start seed
    key = jax.random.split(jax.random.PRNGKey(key),10*epochs)
    k = 0

    #If video, create dir to save images
    if video:
        os.system('rm -r tmp_' + filename)
        os.system('mkdir tmp_' + filename)
        dir = '/tmp_' + filename
        os.chdir(os.getcwd() + dir)

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
        intervals = -1 + np.zeros((1,d)) #Array with intervals
        block = np.array([0]) #Array with block of each interval

    #Store error
    current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,key[k,0],num_classes) #Get error
    k = k + 1 #Update seed
    best_error = current_error #Best error_batch
    best_intervals = intervals.copy() #Best intervals
    best_block = block.copy() #Best block

    #If video
    if video:
        dt.picture_partition(intervals,block,title = 'Epoch 0 Error = ' + str(round(current_error,3)),filename = filename + '_' + str(0).zfill(5))

    #Objects to trace
    trace_error = np.array([]) #Trace algorithm time
    trace_time = np.array([]) #Trace algorithm error
    time_unite = np.array([])
    time_dismenber = np.array([])
    time_break = np.array([])

    #For each epoch
    print('- Starting epochs')
    tinit = time.time() #Initialize time
    print(' Initial error: ' + str(round(best_error,3))) #Prits initial error
    with alive_bar(epochs) as bar: #Alive bar for tracing
        for e in range(epochs): #For each epoch
            if batches > 1: #If there should be training batches
                if unique: #If data is unique, batches of frequency table
                    tab_train = np.random.permutation(tab_train,0) #Random permutation of training table
                    k = k + 1 #Update seed
                    tab_val = np.random.permutation(tab_val,0) #Random permutation of validation table
                    k = k + 1 #Update seed
                else: #Batches of data
                    train = np.random.permutation(train,0) #Random permutation of training data
                    k = k + 1 #Update seed
                    val = np.random.permutation(val,0) #Random permutation of validation data
                    k = k + 1 #Update seed
            for b in range(batches): #For each batch
                #Get frequency table of batch
                tab_train_batch,tab_val_batch,bnval = ut.get_tfrequency_batch(b,batches,tab_train,tab_val,train,val,bsize,bsize_val,unique,batch_val,nval,num_classes)

                #Compute probabilities
                small = np.array(math.comb(np.max(block) + 1,2)) #Number of ways of uniting blocks
                dismenber = np.power(np.bincount(block) - 1,2) - 1 #Number of ways of dimenbering
                dismenber = np.where(dismenber == -1,0,dismenber)
                break_int = np.array(ut.count_points(intervals))
                prob = np.append(np.append(small,np.sum(dismenber)),np.sum(break_int)) #Probability of uniting, diemenbering and breaking interval al internal point
                what_nei = np.random.choice(np.array([0,1,2]),size=(sample,),p = prob/np.sum(prob)) #Sample kind of step to take at each sample neighbor
                k = k + 1 #Update seed
                break_int = break_int/np.sum(break_int)
                if jnp.sum(dismenber) > 0:
                    dismenber = dismenber/np.sum(dismenber)
                #break_int[-1] = 1 - jnp.sum(break_int[:-1])

                #Objects to store neighbors
                store_nei = list() #Store neighbors
                error_batch = np.array([]) #Store error

                #Sample neighbors
                for n in range(sample): #For each neighbor
                    if what_nei[n] == 0: #Unite intervals
                        #Sample block to unite intervals
                        unite = np.random.choice(np.arange(np.max(block) + 1),size=(2,),replace = False)
                        k = k + 1 #Update seed

                        #Sample intervals to unite in the sampled block and store the result
                        store_nei.append(ut.unite_blocks(unite,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 1: #Dismenber intervals
                        #Sample block to dismenber intervals
                        b_dis = np.random.choice(np.arange(np.max(block) + 1),size=(1,),p = dismenber)
                        k = k + 1 #Update seed

                        #Sample dismenbering of the sampled block and store the result
                        store_nei.append(ut.dismenber_block(b_dis,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 2: #Break interval
                        #Sample interval to break
                        interval_break = np.random.choice(np.arange(intervals.shape[0]),size=(1,),p = break_int)
                        k = k + 1 #Update seed

                        #Break interval on sampled point and store the result
                        store_nei.append(ut.break_interval(interval_break,intervals[interval_break,:].copy(),block[interval_break].copy(),intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = key[k,0],num_classes = num_classes))
                        k = k + 1 #Update seed

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                #Update partition at each batch
                which_nei = np.where(error_batch == np.min(error_batch))[0][0] #Get first neighbor with the least error
                block = store_nei[which_nei]['block'].copy() #Update block
                intervals = store_nei[which_nei]['intervals'].copy() #Update interval
                del store_nei, error_batch #Delete trace of neighbors
            #Get error current partition at the end of epoch
            current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,key[k,0],num_classes)
            k = k + 1 #Update seed

            #Store current partition as best with it has the least error so far
            if current_error < best_error:
                best_error = current_error #Store error
                best_intervals = intervals.copy() #Store intervals
                best_block = block.copy() #Store blocks
                print('Time: ' + str(round(time.time() - tinit,2)) + ' Error: ' + str(round(best_error,3))) #Print error

            #Trace
            trace_error = np.append(trace_error,current_error) #Trace error
            trace_time = np.append(trace_time,np.array([time.time() - tinit])) #Trace time

            #If video
            if video:
                dt.picture_partition(intervals,block,title = 'Epoch ' + str(e) + ' Error = ' + str(round(current_error,3)),filename = filename + '_' + str(e + 1).zfill(5))
            bar() #Update bar
    #Test error
    test_error = None #Initialize test error
    if test is not None: #Compute test error if there is test data
        test_error = ut.error_partition(tab_train,tab_test,intervals,block,test.shape[0],key[k,0],num_classes)
        k = k + 1

    #Create video
    if video:
        os.system('for f in *.pdf; do convert -density 500 ./"$f" -quality 100 -background white -alpha remove -alpha off ./"${f%.pdf}.png"; done')
        os.system("ffmpeg -framerate " + str(framerate) + " -i " + filename + "_%5d.png " + filename + ".mp4")

    #Estimated function
    label_intervals = ut.estimate_label_partition(tab_train,best_intervals,best_block,num_classes,key = key[k,0])
    f = ut.get_estimated_function(tab_train,best_intervals,best_block,num_classes,key = key[k,0])

    #Return
    return {'block': best_block,'intervals': best_intervals,'best_error': best_error,'test_error': test_error,'trace_error': trace_error,'trace_time': trace_time,'label_intervals': label_intervals,'f': f}

#Stochastic ISI algorithm
def stochasticISI(train,class_break,test = None,num_classes = 2,key = 0,unique = False,intervals = None,block = None):
    """
    Stochastic Incremental Splitting of Intervals (ISI) algorithm
     -------
    Parameters
    ----------
    train : jax.numpy.array

        Array with training data. The last column contains the labels

    class_break : int

        Which class to break the intervals on

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
    #Start seed
    key = jax.random.split(jax.random.PRNGKey(key),10*train.shape[0])
    k = 0

    #Parameters
    d = train.shape[1] - 1
    trace_error = np.array([])
    trace_time = np.array([])

    #Frequency table
    tab_train = dt.get_ftable(train,unique,num_classes)
    tab_train = jax.random.permutation(jax.random.PRNGKey(key[k,0]),tab_train,0) #Random permutation
    ntrain = train.shape[0]
    k = k + 1
    if test is not None:
        tab_test = dt.get_ftable(test,unique,num_classes)

    #Get label of each observed domain point
    label = jax.vmap(lambda x: np.where(x == np.max(x),1,0))(tab_train[:,-num_classes:])

    #Only points without tie
    tab = tab_train[np.sum(label,1) == 1,:]
    label = label[np.sum(label,1) == 1,:]
    label = np.sum(jax.vmap(lambda x: np.where(x == 1,np.arange(num_classes),0))(label),1)

    #Initialize interval
    if intervals is None or block is None:
        intervals = -1 + np.zeros((1,d)) #Array with intervals
        block = np.array([0]) #Array with block of each interval

    #Probability of each point
    t0 = time.time()
    freq = jax.vmap(lambda interval: ut.frequency_labels_interval(interval,tab[:,:-num_classes],label,num_classes))(intervals)
    prob = np.where((freq[:,class_break] > 0)*(np.max(np.delete(freq,class_break,1),1) > 0),1,0)
    prob = np.sum(jax.vmap(lambda point: np.where(ut.get_interval(point,intervals),prob,0))(tab[:,:-num_classes]),1)
    prob = np.where(label != class_break,0,prob)
    limit = ut.get_limits_some_interval(intervals,tab[:,:-num_classes])
    prob = np.where(limit,0,prob)
    step = 0
    best_error = 1
    with alive_bar() as bar:
        while(np.max(prob) == 1):
            #Sample point to break
            point_break = tab[jax.random.choice(jax.random.PRNGKey(key[k,0]), np.array(list(range(tab.shape[0]))),shape=(1,),p = prob/np.sum(prob)),:]
            k = k + 1 #Update seed
            #Update partition
            new_partition = ut.break_interval(point_break,intervals,block,ntrain,tab_train,tab_train,step = True,key = key[k,0],num_classes = num_classes)
            intervals = new_partition['intervals']
            block = new_partition['block']
            error = new_partition['error']
            k = k + 1 #Update seed
            #Update probabilities
            freq = jax.vmap(lambda interval: ut.frequency_labels_interval(interval,tab[:,:-num_classes],label,num_classes))(intervals)
            prob = np.where((freq[:,class_break] > 0)*(np.max(np.delete(freq,class_break,1),1) > 0),1,0)
            prob = np.sum(jax.vmap(lambda point: np.where(ut.get_interval(point,intervals),prob,0))(tab[:,:-num_classes]),1)
            prob = np.where(label != class_break,0,prob)
            limit = ut.get_limits_some_interval(intervals,tab[:,:-num_classes])
            prob = np.where(limit,0,prob)
            step = step + 1
            trace_error = np.append(trace_error,error)
            trace_time = np.append(trace_time,np.array(time.time() - t0))
            if error < best_error:
                print('Step ' + str(step) + ' Error ' + str(round(error,3)))
                best_error = error
            bar()

    #Get test error
    test_error = None
    if test is not None:
        test_error = ut.error_partition(tab,tab_test,intervals,block,test.shape[0],key[k,0],num_classes)
        k = k + 1

    #Estimated function
    label_intervals = ut.estimate_label_partition(tab_train,intervals,block,num_classes,key = key[k,0])
    f = ut.get_estimated_function(tab_train,intervals,block,num_classes,key = key[k,0])

    return {'block': block,'intervals': intervals,'test_error': test_error,'trace_error': trace_error,'trace_time': trace_time,'label_intervals': label_intervals,'f': f}
