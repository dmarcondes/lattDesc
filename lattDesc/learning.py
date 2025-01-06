#Lattice descent on the Interval Parition Lattice
#import jax
#jax.config.update('jax_platform_name', 'cpu')
#from jax import numpy as jnp
from lattDesc import data as dt
from lattDesc import utils as ut
import numpy as np
import math
import time
from alive_progress import alive_bar
import os

#Stochastic Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(train,val,epochs = 10,sample = 10,projection = True,batches = 1,batch_val = False,test = None,num_classes = 2,key = 0,unique = False,intervals = None,block = None,video = False,filename = 'video_sdesc_BIPL',framerate = 1):
    """
    Stochastic Lattice Descent Algorithm in the Boolean Interval Partition Lattice
    -------
    Parameters
    ----------
    train : numpy.array

        Array with training data. The last column contains the labels

    val : numpy.array

        Array with validation data. The last column contains the labels

    epochs : int

        Training epochs

    sample : int

        Number of neighbors to sample at each step

    projection = logical

        Whether to search a projection of the lattice on the data (True) or the whole lattice

    batches : int

        Number of sample batches in each epoch

    batch_val : logical

        Whether to consider batches for the validation data

    test : numpy.array

        Array with test data. The last column contains the labels

    num_classes : int

        Number of classes

    key : int

        Seed for sampling

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    intervals : numpy.array

        Array of initial intervals. If None then initialize with the unitary partition

    block : numpy.array

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
    rng = np.random.default_rng(seed = key)

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

    #Projection
    if not projection and d > 25:
        projection = True
        print('\n Warning: Projection is being forced since d > 25 variables \n')

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
    current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,int(rng.choice(np.arange(1e6))),num_classes) #Get error
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
    print(' Initial error: ' + str(round(best_error,3))) #Prints initial error
    with alive_bar(epochs) as bar: #Alive bar for tracing
        for e in range(epochs): #For each epoch
            if batches > 1: #If there should be training batches
                if unique: #If data is unique, batches of frequency table
                    tab_train = rng.permutation(tab_train,0) #Random permutation of training table
                    tab_val = rng.permutation(tab_val,0) #Random permutation of validation table
                else: #Batches of data
                    train = rng.permutation(train,0) #Random permutation of training data
                    val = rng.permutation(val,0) #Random permutation of validation data
            for b in range(batches): #For each batch
                #Get frequency table of batch
                tab_train_batch,tab_val_batch,bnval = ut.get_tfrequency_batch(b,batches,tab_train,tab_val,train,val,bsize,bsize_val,unique,batch_val,nval,num_classes)

                #Compute probabilities
                small = np.array(math.comb(np.max(block) + 1,2)) #Number of ways of uniting blocks
                dismenber = np.power(np.bincount(block) - 1,2) - 1 #Number of ways of dimenbering
                dismenber = np.where(dismenber == -1,0,dismenber)
                if projection:
                    break_int = np.array(1 - ut.get_limits_some_interval(intervals,tab_train_batch[:,0:-num_classes]))
                else:
                    break_int = np.array(ut.count_points(intervals))
                prob = np.append(np.append(small,np.sum(dismenber)),np.sum(break_int)) #Probability of uniting, diemenbering and breaking interval al internal point
                what_nei = rng.choice(np.array([0,1,2]),size=(sample,),p = prob/np.sum(prob)) #Sample kind of step to take at each sample neighbor
                break_int = break_int/np.sum(break_int)
                if np.sum(dismenber) > 0:
                    dismenber = dismenber/np.sum(dismenber)

                #Objects to store neighbors
                store_nei = list() #Store neighbors
                error_batch = np.array([]) #Store error

                #Sample neighbors
                for n in range(sample): #For each neighbor
                    if what_nei[n] == 0: #Unite intervals
                        #Sample block to unite intervals
                        unite = rng.choice(np.arange(np.max(block) + 1),size=(2,),replace = False)

                        #Sample intervals to unite in the sampled block and store the result
                        store_nei.append(ut.unite_blocks(unite,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = int(rng.choice(np.arange(1e6))),num_classes = num_classes))

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 1: #Dismenber intervals
                        #Sample block to dismenber intervals
                        b_dis = rng.choice(np.arange(np.max(block) + 1),size=(1,),p = dismenber)

                        #Sample dismenbering of the sampled block and store the result
                        store_nei.append(ut.dismenber_block(b_dis,intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = int(rng.choice(np.arange(1e6))),num_classes = num_classes))

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                    elif what_nei[n] == 2: #Break interval
                        if projection:
                            #Sample point to break on
                            point_break = tab_train_batch[rng.choice(np.arange(tab_train_batch.shape[0]),size = (1,),p = break_int),:-num_classes]
                            interval_break = np.where(ut.get_interval(point_break,intervals))[0]
                        else:
                            #Sample interval to break
                            interval_break = rng.choice(np.arange(intervals.shape[0]),size=(1,),p = break_int)
                            point_break = None

                        #Break interval on sampled point and store the result
                        store_nei.append(ut.break_interval(point_break,interval_break,intervals[interval_break,:].copy(),block[interval_break].copy(),intervals.copy(),block.copy(),bnval,tab_train_batch,tab_val_batch,step = True,key = int(rng.choice(np.arange(1e6))),num_classes = num_classes))

                        #Store error
                        error_batch = np.append(error_batch,store_nei[-1]['error'])
                #Update partition at each batch
                which_nei = np.where(error_batch == np.min(error_batch))[0][0] #Get first neighbor with the least error
                block = store_nei[which_nei]['block'].copy() #Update block
                intervals = store_nei[which_nei]['intervals'].copy() #Update interval
                del store_nei, error_batch #Delete trace of neighbors
            #Get error current partition at the end of epoch
            current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,int(rng.choice(np.arange(1e6))),num_classes)

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
        test_error = ut.error_partition(tab_train,tab_test,intervals,block,test.shape[0],int(rng.choice(np.arange(1e6))),num_classes)

    #Create video
    if video:
        os.system('for f in *.pdf; do convert -density 500 ./"$f" -quality 100 -background white -alpha remove -alpha off ./"${f%.pdf}.png"; done')
        os.system("ffmpeg -framerate " + str(framerate) + " -i " + filename + "_%5d.png " + filename + ".mp4")

    #Estimated function
    k = int(rng.choice(np.arange(1e6)))
    label_intervals = ut.estimate_label_partition(tab_train,best_intervals,best_block,num_classes,key = k)
    f = ut.get_estimated_function(tab_train,best_intervals,best_block,num_classes,key = k)

    #Return
    return {'block': best_block,'intervals': best_intervals,'best_error': best_error,'test_error': test_error,'trace_error': trace_error,'trace_time': trace_time,'label_intervals': label_intervals,'f': f}

#ISI algorithm
def ISI(train,intervals = None,unique = False,key = 0):
    """
    Incremental Splitting of Intervals (ISI)
    -------
    Parameters
    ----------
    train : numpy.array

        Array with training data. The last column contains the labels

    intervals : numpy.array

        Initial intervals

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    key : int

        Seed for sampling

    Returns
    -------
    dictionary with the learned 'basis' and function 'f'
    """
    print('------Starting algorithm------')
    #Start seed
    rng = np.random.default_rng(seed = key)

    #Get frequency tables
    print('- Creating frequency tables')
    d = train.shape[1] - 1 #dimension
    tab_train = dt.get_ftable(train,unique,2) #Training table

    #Get zero and one points
    zero_points = rng.permutation(tab_train[np.where(tab_train[:,d] > tab_train[:,d + 1])[0],:d])
    one_points = tab_train[np.where(tab_train[:,d] < tab_train[:,d + 1])[0],:d]

    #Initialize intervals
    if intervals is None:
        intervals = -1 + np.zeros((1,d))

    #Break intervals in sequence
    print('- Running...')
    with alive_bar(zero_points.shape[0]) as bar: #Alive bar for tracing
        for i in range(zero_points.shape[0]):
            point = zero_points[i,:]
            #Get intervals that contain point
            which_interval = ut.get_interval(point,intervals)
            print(intervals.shape[0])
            del_intervals = np.delete(intervals,np.where(which_interval),0)
            if np.sum(which_interval) > 0:
                for k in np.where(which_interval)[0]:
                    #Get break interval
                    break_interval = intervals[k,:]
                    #Limits of interval
                    A = np.where(break_interval == -1,0,break_interval)
                    B = np.where(break_interval == -1,1,break_interval)
                    #Add intervals
                    for j in np.where(np.logical_and(point == 1,A == 0))[0]:
                        x = np.zeros((1,d))
                        x[0,j] = 1
                        x = 1 - x
                        B_tmp = np.minimum(B,x)
                        interval_tmp = np.where((A == 0)*(B_tmp == 1),-1,A)
                        if del_intervals.shape[0] > 0:
                            if not ut.contained_some(interval_tmp,del_intervals):
                                intervals = np.append(intervals,interval_tmp,0)
                        else:
                            intervals = np.append(intervals,interval_tmp,0)
                    for j in np.where(np.logical_and(point == 0,B == 1))[0]:
                        x = np.zeros((1,d))
                        x[0,j] = 1
                        A_tmp = np.maximum(A,x)
                        interval_tmp = np.where((A_tmp == 0)*(B == 1),-1,B)
                        if del_intervals.shape[0] > 0:
                            if not ut.contained_some(interval_tmp,del_intervals):
                                intervals = np.append(intervals,interval_tmp,0)
                        else:
                            intervals = np.append(intervals,interval_tmp,0)
                intervals = np.delete(intervals,np.where(which_interval)[0],0)
                #Erase intervals that do not contain one points
                intervals = np.delete(intervals,np.where(np.sum(ut.get_elements_each_interval(intervals,one_points),1) == 0)[0],0)
            bar()

    #Get estimated function
    f = lambda x: int(ut.get_elements_some_interval(intervals,x.reshape((1,d)))[0])

    return {'basis': intervals,'f': f}
