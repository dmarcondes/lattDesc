#Lattice descent on the Interval Parition Lattice
from lattDesc import data as dt
from lattDesc import utils as ut
import numpy as np
import math
import time
from alive_progress import alive_bar
import os
from joblib import Parallel, delayed

#Empirical risk minimization
def ERM(train = None,test= None,unique = False,num_classes = 2,tab_train = None,tab_test = None):
    """
    Learning by empirical risk minimization
    -------
    Parameters
    ----------
    train,test : numpy.array

        Arrays with training and test data. The last column contains the labels. Only necessary if the training or test frequency table are not provided

    tab_train,tab_test : numpy.array

        Array with the frequency table of training and test data. Optional

    unique : logical

        Whether the training and test data are unique, i.e., each input point appears at most once in each dataset

    num_classes : int

        Number of classes

    Returns
    -------
    dictionary with the learned function and the test classification error
    """
    #Table
    if tab_train is None:
        tab_train = dt.get_ftable(train,unique,num_classes)
    #Function for predicting
    def f(x):
        x_freq = tab_train[np.apply_along_axis(lambda z: (x == z).all(),1,tab_train[:,:-num_classes]),-num_classes:]
        if x_freq.shape[0] > 0:
            return np.random.choice(np.where((x_freq == np.max(x_freq))[0,:])[0],1)
        else:
            return -1
    #Test error
    error = None
    if tab_test is not None:
        pred = np.apply_along_axis(f,1,tab_test[:,:-num_classes]).reshape((tab_test.shape[0],))
        c = np.array(range(num_classes))
        freq = tab_test[:,-num_classes:]
        error = np.sum(np.apply_along_axis(lambda i: np.sum(freq[i,c != pred[i]]),1,np.arange(tab_test.shape[0]).reshape((tab_test.shape[0],1))))/np.sum(freq)
    elif test is not None:
        error = np.sum(np.apply_along_axis(f,1,test[:,:-1]) != test[:,-1])/test.shape[0]
    return {'f': f,'test_error': error}

#Stochastic Lattice Descent on the Boolean Interval Partition Lattice
def sdesc_BIPL(train = None,val = None,test = None,tab_train = None,tab_val = None,tab_test = None,epochs = 10,sample = 10,batches = 1,batch_val = False,n_jobs = 8,num_classes = 2,key = 0,unique = False,intervals = None,block = None,video = False,filename = 'video_sdesc_BIPL',framerate = 1):
    """
    Stochastic Lattice Descent Algorithm in the Boolean Interval Partition Lattice
    -------
    Parameters
    ----------
    train,val,test : numpy.array

        Arrays with training, validation and test data. The last column contains the labels. Only necessary if the training, validation or test frequency table are not provided

    tab_train,tab_val,tab_test : numpy.array

        Array with the frequency table of training, validation and test data. Optional

    epochs : int

        Training epochs

    sample : int

        Number of neighbors to sample at each step

    batches : int

        Number of sample batches in each epoch

    batch_val : logical

        Whether to consider batches for the validation data

    n_jobs : int

        Number of parallel jobs

    num_classes : int

        Number of classes

    key : int

        Seed for sampling

    unique : logical

        Whether the data is unique, i.e., each input point appears at most once in each dataset data

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
    dictionary with the learned 'block','intervals','label_intervals','best_error', 'test_error' and the estimated function ('f'), and the trace of the error ('trace_error') and time ('trace_time') over the epochs
    """
    print('------Starting Lattice Descent Algorithm in the BIPL------')
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
    if tab_train is None:
        tab_train = dt.get_ftable(train,unique,num_classes) #Training table
    d = tab_train.shape[1] - num_classes #Input dimension
    ntrain = np.sum(tab_train[:,-num_classes:]) #Training sample size
    if tab_val is None:
        tab_val = dt.get_ftable(val,unique,num_classes) #Validation table
    nval = np.sum(tab_val[:,-num_classes:]) #Validation sample size
    if tab_test is None and test is not None: #Get test table if there is test data
        tab_test = dt.get_ftable(test,unique,num_classes)

    #Batch size
    bsize = math.floor(ntrain/batches)
    bsize_val = math.floor(nval/batches)

    #Initial partition
    print('- Initializing objects')
    if intervals is None or block is None: #If initial partition is not given, initialize
        intervals = -1 + np.zeros((1,d)) #Array with intervals
        block = np.array([0]) #Array with block of each interval

    #Store error
    current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes) #Get error
    best_error = current_error #Best error_batch
    best_intervals = intervals.copy() #Best intervals
    best_block = block.copy() #Best block

    #If video
    if video:
        dt.picture_partition(intervals,block,title = 'Epoch 0 Error = ' + str(round(current_error,3)),filename = filename + '_' + str(0).zfill(5))

    #Objects to trace
    trace_error = np.array([]) #Trace algorithm time
    trace_time = np.array([]) #Trace algorithm error

    #For each epoch
    print('- Starting epochs')
    tinit = time.time() #Initialize time
    print(' Initial error: ' + str(round(best_error,3))) #Prints initial error
    with alive_bar(epochs) as bar: #Alive bar for tracing
        for e in range(epochs): #For each epoch
            if batches > 1:
                tab_train_epoch = tab_train.copy()
                tab_val_epoch = tab_val.copy()
            else:
                tab_train_batch = tab_train.copy()
                tab_val_batch = tab_val.copy()
            bnval = nval
            for b in range(batches): #For each batch
                #Get frequency table of batch
                if batches > 1:
                    tab_train_epoch,tab_train_batch,tab_val_epoch,tab_val_batch,bnval = ut.get_tfrequency_batch(tab_train_epoch,tab_val_epoch,bsize,bsize_val,batch_val,nval,rng,num_classes)

                #Compute probabilities
                total_unite = np.array(math.comb(np.max(block) + 1,2)).astype('float64') #Number of ways of uniting blocks
                break_points = np.array(1 - ut.get_limits_some_interval(intervals,tab_train_batch[:,0:-num_classes])) #Points on which intervals can be broken
                total_break = np.sum(break_points).astype('float64') #Number of points on which intervals can be broken (internal points)
                dismember = np.power(np.bincount(block) - 1,2).astype('float64') - 1 #Number of ways of dismemebering blocks for each blocks
                dismember[dismember < 0 ] = 0 #Correct blocks with only one interval
                total_dismember = np.sum(dismember) #Total number of ways of dismemebering blocks
                total = total_unite + total_break + total_dismember #Total manners of obtaining neighbors
                prob_manners = np.array([total_unite/total,total_dismember/total,total_break/total]).reshape((3,)) #Probabilities of uniting, dismembering and breaking interval at internal point

                #Sample kind of step at each neighbor
                kind_nei = rng.choice(np.array([0,1,2]),size=(sample,),p = prob_manners) #Sample kind of step to take at each sample neighbo

                #Objects to store neighbors
                error_batch = np.array([]) #Store error

                #Sample neighbors
                if total_dismember == 0:
                    total_dismember = 1
                if total_break == 0:
                    total_break = 1
                n_key = np.round(rng.uniform(0,100000,(sample,))).astype(np.int32)
                process = lambda n: ut.sample_visit_neighbor(kind_nei[n],dismember/total_dismember,break_points/total_break,block,intervals,bnval,tab_train_batch,tab_val_batch,np.random.default_rng(seed = n_key[n]),num_classes)
                store_nei = Parallel(n_jobs = n_jobs)(delayed(process)(n) for n in range(sample))
                for i in range(len(store_nei)):
                    error_batch = np.append(error_batch,store_nei[i]['error'])

                #Update partition at each batch
                which_nei = np.where(error_batch == np.min(error_batch))[0][0] #Get first neighbor with the least error
                block = store_nei[which_nei]['block'].copy() #Update block
                intervals = store_nei[which_nei]['intervals'].copy() #Update interval
                del store_nei, error_batch #Delete trace of neighbors
            #Get error current partition at the end of epoch
            current_error = ut.error_partition(tab_train,tab_val,intervals,block,nval,rng,num_classes)

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
    if tab_test is not None: #Compute test error if there is test data
        test_error = ut.error_partition(tab_train,tab_test,intervals,block,np.sum(tab_test[:,-num_classes:]),rng,num_classes)

    #Create video
    if video:
        os.system('for f in *.pdf; do convert -density 500 ./"$f" -quality 100 -background white -alpha remove -alpha off ./"${f%.pdf}.png"; done')
        os.system("ffmpeg -framerate " + str(framerate) + " -i " + filename + "_%5d.png " + filename + ".mp4")

    #Estimated function
    label_intervals = ut.estimate_label_partition(tab_train,best_intervals,best_block,rng,num_classes)
    f = ut.get_estimated_function(tab_train,best_intervals,best_block,rng,num_classes)

    #Return
    return {'block': best_block,'intervals': best_intervals,'best_error': best_error,'test_error': test_error,'trace_error': trace_error,'trace_time': trace_time,'label_intervals': label_intervals,'f': f}

#ISI algorithm
def ISI(train = None,test = None,tab_train = None,tab_test = None,intervals = None,unique = False,key = 0):
    """
    Incremental Splitting of Intervals (ISI)
    -------
    Parameters
    ----------
    train,test : numpy.array

        Arrays with training and test data. The last column contains the labels. Only necessary if the training or test frequency table are not provided

    tab_train,tab_test : numpy.array

        Array with the frequency table of training and test data. Optional

    intervals : numpy.array

        Array of initial intervals. If not provided, initiate from the whole Boolean lattice

    unique : logical

        Whether the data is unique, i.e., each input point appears at most once in the datasets

    key : int

        Seed for sampling

    Returns
    -------
    dictionary with the learned 'basis', function 'f', 'test_error' and 'total_time'
    """
    print('------Starting ISI algorithm------')
    #Start seed
    rng = np.random.default_rng(seed = key)

    #Get frequency tables
    if tab_train is None:
        print('- Creating frequency tables')
        tab_train = dt.get_ftable(train,unique,2) #Training table
    d = tab_train.shape[1] - 2 #dimension

    #Get zero and one points
    zero_points = rng.permutation(tab_train[np.where(tab_train[:,d] > tab_train[:,d + 1])[0],:d])
    one_points = tab_train[np.where(tab_train[:,d] < tab_train[:,d + 1])[0],:d]

    #Initialize intervals
    if intervals is None:
        intervals = -1 + np.zeros((1,d))

    #Break intervals in sequence
    print('- Running...')
    tinit = time.time()
    with alive_bar(zero_points.shape[0]) as bar: #Alive bar for tracing
        for i in range(zero_points.shape[0]): #For each zero point
            point = zero_points[i,:] #Get points
            #Get intervals that contain point
            which_interval = ut.get_interval(point,intervals)
            if np.sum(which_interval) > 0: #If there is an interval that contains the point
                #Delete these intervals
                del_intervals = np.delete(intervals,np.where(which_interval),0)
                for k in np.where(which_interval)[0]: #For each interval that contains the point
                    #Get the interval
                    break_interval = intervals[k,:]
                    #Get limits of interval
                    A = np.where(break_interval == -1,0,break_interval)
                    B = np.where(break_interval == -1,1,break_interval)
                    #Get intervals obtained by breaking on the point
                    for j in np.where(np.logical_and(point == 1,A == 0))[0]: #Intervals with limit A
                        x = np.zeros((1,d))
                        x[0,j] = 1
                        x = 1 - x
                        B_tmp = np.minimum(B,x)
                        interval_tmp = np.where((A == 0)*(B_tmp == 1),-1,A)
                        if del_intervals.shape[0] > 0:
                            if not ut.contained_some(interval_tmp[0,:],del_intervals): #Test if new interval is maximal
                                intervals = np.append(intervals,interval_tmp,0)
                        else:
                            intervals = np.append(intervals,interval_tmp,0)
                    for j in np.where(np.logical_and(point == 0,B == 1))[0]: #Intervals with limit B
                        x = np.zeros((1,d))
                        x[0,j] = 1
                        A_tmp = np.maximum(A,x)
                        interval_tmp = np.where((A_tmp == 0)*(B == 1),-1,B)
                        if del_intervals.shape[0] > 0: #Test if new interval is maximal
                            if not ut.contained_some(interval_tmp[0,:],del_intervals):
                                intervals = np.append(intervals,interval_tmp,0)
                        else:
                            intervals = np.append(intervals,interval_tmp,0)
                intervals = np.delete(intervals,np.where(which_interval)[0],0) #Delete intervals that contain point
                #Erase intervals that do not contain one points
                intervals = np.delete(intervals,np.where(np.sum(ut.get_elements_each_interval(intervals,one_points),1) == 0)[0],0)
            bar()
    total_time = time.time() - tinit

    #Get estimated function
    f = lambda data: (ut.get_elements_some_interval(intervals,data)).astype('int32')

    #Test error
    test_error = None #Initialize test error
    if test is not None or tab_test is not None: #Compute test error if there is test data
        if tab_test is not None:
            pred = f(tab_test[:,:-2]).reshape((tab_test.shape[0],))
            c = np.array(range(2))
            freq = tab_test[:,-2:]
            test_error = np.sum(np.apply_along_axis(lambda i: np.sum(freq[i,c != pred[i]]),1,np.arange(tab_test.shape[0]).reshape((tab_test.shape[0],1))))/np.sum(freq)
        else:
            pred = f(test[:,:-1])
            test_error = np.sum(np.abs(np.array(pred) - test[:,-1]))/test.shape[0]

    return {'basis': intervals,'f': f,'test_error': test_error,'total_time': total_time}

#Disjoint ISI algorithm
def disjoint_ISI(train = None,tab_train = None,intervals = None,block = None,unique = False,key = 0):
    """
    Disjoint Incremental Splitting of Intervals (ISI)
    -------
    Parameters
    ----------
    train : numpy.array

        Array with training data. The last column contains the labels. Only necessary if the training frequency table is not provided

    tab_train : numpy.array

        Array with the frequency table of training data. Optional

    intervals : numpy.array

        Array of initial intervals. If not provided, initiate from the unitary partition

    block : numpy.array

        Initial blocks. If not provided, initiate from the unitary partition

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    key : int

        Seed for sampling

    Returns
    -------
    dictionary with the final 'intervals', 'block', 'train_error','steps','total_time' and estimated function ('f')
    """
    print('------Starting algorithm------')
    #Start seed
    rng = np.random.default_rng(seed = key)

    #Get frequency tables
    if tab_train is None:
        print('- Creating frequency tables')
        tab_train = dt.get_ftable(train,unique,2) #Training table
    d = tab_train.shape[1] - 2
    ntrain = np.sum(tab_train[:,-2:])

    #Get zero and one points
    zero_points = rng.permutation(tab_train[np.where(tab_train[:,d] > tab_train[:,d + 1])[0],:d])
    one_points = tab_train[np.where(tab_train[:,d] < tab_train[:,d + 1])[0],:d]

    #Initialize intervals
    if intervals is None:
        intervals = -1 + np.zeros((1,d))
        block = np.array([0])

    #Initialize step
    step = 1

    #Probability of breaking each interval
    pzero = np.sum(np.logical_and(ut.get_elements_each_interval(intervals,zero_points),1 - ut.get_limits_each_interval(intervals,zero_points)),1) #Zero points in each interval that are not limits
    pone = np.sum(np.logical_and(ut.get_elements_each_interval(intervals,one_points),1 - ut.get_limits_each_interval(intervals,one_points)),1) #One points in each interval that are not limits
    prob = np.where(pone == 0,0,pzero) + np.where(pzero == 0,0,pone) #Number of points in each interval that contains both zero and one points that are not limits
    print('- Running...')
    tinit = time.time()
    while(np.sum(prob) > 0): #While there are intervals containing both zero and one points that are not limits
        print('Step ' + str(step) + ' Time: ' + str(np.round(time.time() - tinit,2)))
        #Normalize probability
        prob = prob.astype('float64')
        prob = prob/np.sum(prob)
        #Sample interval to break
        i = rng.choice(np.arange(intervals.shape[0]),size = 1,p = prob)
        #Sample point in interval to break on
        log = np.logical_and(1 - ut.get_limits_interval(intervals[i,:],tab_train[:,:d]),ut.get_elements_some_interval(intervals[i,:],tab_train[:,:d])) #Flag points in interval
        k = rng.choice(np.where(log)[0],size = 1) #Sample point in interval
        #Break interval
        point_break = tab_train[k,:d]
        res_break = ut.break_interval(point_break,i,intervals[i,:].copy(),block[i].copy(),intervals.copy(),block.copy(),ntrain,tab_train,tab_train,step = True,rng = rng,num_classes = 2,compute_error = False)
        intervals = res_break['intervals'].copy()
        block = ut.estimate_label_partition(tab_train,intervals,np.arange(intervals.shape[0]),rng,num_classes = 2).astype(np.int32)
        if np.min(block) == 1:
            block = block - 1
        #Reduce one block
        if np.sum(block == 1) > 1:
            reduced = False
        else:
            reduced = True
        intervals_one = intervals[block == 1,:]
        while(not reduced):
            intervals_one,reduced = ut.reduce(intervals_one)
        #Reduce zero block
        if np.sum(block == 0) > 1:
            reduced = False
        else:
            reduced = True
        intervals_zero = intervals[block == 0,:]
        while(not reduced):
            intervals_zero,reduced = ut.reduce(intervals_zero)
        #Update intervals and blocks as reduced
        intervals = np.append(intervals_one,intervals_zero,0)
        block = np.append(1 + np.zeros((intervals_one.shape[0],)),np.zeros((intervals_zero.shape[0],))).astype(np.int32)
        if np.min(block) == 1:
            block = block - 1
        #Probability of each interval
        pzero = np.sum(np.logical_and(ut.get_elements_each_interval(intervals,zero_points),1 - ut.get_limits_each_interval(intervals,zero_points)),1)
        pone = np.sum(np.logical_and(ut.get_elements_each_interval(intervals,one_points),1 - ut.get_limits_each_interval(intervals,one_points)),1)
        prob = np.where(pone == 0,0,pzero) + np.where(pzero == 0,0,pone)
        #Update step
        step = step + 1
    total_time = time.time() - tinit
    error = ut.error_partition(tab_train,tab_train,intervals,block,ntrain,rng,num_classes = 2)

    return {'intervals': intervals,'block': block,'train_error': error,'step': step - 1,'f': ut.get_estimated_function(tab_train,intervals,block,rng,num_classes = 2),'total_time': total_time}
