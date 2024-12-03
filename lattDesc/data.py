#Process data for lattice descent algorithms
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
import pandas as pd
from itertools import product

#Generate binary synthetic data
def synthetic_binary_data(n,d,key):
    """
    Generate synthetic binary random data
    -------
    Parameters
    ----------
    n : int

        Sample size

    d : int

        Dimension of the input

    key : int

        Seed for sampling

    Returns
    -------
    jax.numpy.array withe genrated data in which the last column constains the labels
    """
    #Generate probability of each point
    key = jax.random.split(jax.random.PRNGKey(key),n+3)
    prob = jax.random.uniform(key = jax.random.PRNGKey(key[0,0]),shape = (2 ** d,),minval = 0,maxval = 100)
    prob = prob/jnp.sum(prob)
    #Generate probability of conditioned distribution
    prob_cond = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array([0,1]),shape = (2 ** d,))
    #Generate input points
    index = jax.random.choice(jax.random.PRNGKey(key[2,0]), jnp.array(list(range(2 ** d))), shape=(n,),replace = True,p = prob)
    domain = jnp.array([i for i in product(range(2),repeat = d)])
    x = domain[index,:]
    #Generate output points
    y = []
    for i in range(n):
        p = jnp.array([1 - prob_cond[index[i].tolist()],prob_cond[index[i].tolist()]])
        y = y + [jax.random.choice(jax.random.PRNGKey(key[i + 3,0]), jnp.array(list(range(2))),shape=(1,),p = p).tolist()]
    y = jnp.array(y)
    return jnp.append(x,y,1)

#Get frequency of a point in the domain
def get_fpoint(x,data,num_classes = 2):
    """
    Get empirical frequencies conditioned on a point
    -------
    Parameters
    ----------
    x : jax.numpy.array

        Point to condition on

    data : jax.numpy.array

        Data array

    num_classes : int

        Number of classes

    Returns
    -------
    jax.numpy.array with the empirical frequencies
    """
    data = jax.lax.select(jnp.repeat((data[:,:-1] == x).all(1).reshape((data.shape[0],1)),data.shape[1],1),data,-1 + jnp.zeros(data.shape).astype(data.dtype))
    freq = jax.nn.one_hot(data[:,-1],num_classes)
    return jnp.sum(freq,0)

get_fpoint = jax.jit(get_fpoint,static_argnames = ['num_classes'])

#Get frequence table
def get_ftable(data,unique,num_classes = 2):
    """
    Get empirical frequencies of data with binary outputs
    -------
    Parameters
    ----------
    data : jax numpy array

        Data array

    unique : logical

        Whether the data is unique, i.e., each input point appears only once in the data

    num_classes : int

        Number of classes

    Returns
    -------
    jax.numpy.array
    """
    #Get domain
    if not unique:
        domain = jnp.unique(data[:,:-1],axis = 0)
    else:
        domain = data[:,:-1]
    #Compute frequencies
    f = jax.vmap(lambda x: get_fpoint(x,data,num_classes))(domain)
    return jnp.append(domain,f,1).astype(data.dtype)
