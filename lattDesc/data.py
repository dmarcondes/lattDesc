#Process data for lattice descent algorithms
import jax
from jax import numpy as jnp
import pandas as pd
from itertools import product

#Generate binary synthetic data
def synthetic_binary_data(n,d,key):
    """
    Generate synthetic binary random data.
    -------

    Parameters
    ----------
    n : int

        Sample size

    d : int

        Dimension of the input

    key : int

        Key for sampling

    Returns
    -------

    a JAX numpy array

    """
    #Generate probability of each point
    key = jax.random.split(jax.random.PRNGKey(key),n+3)
    prob = jax.random.uniform(key = jax.random.PRNGKey(key[0,0]),shape = (2 ** d,),minval = 0,maxval = 100)
    prob = prob/jnp.sum(prob)
    #Generate probability of conditioned distribution
    prob_cond = jax.random.choice(jax.random.PRNGKey(key[1,0]), jnp.array([0,1]),shape = (2 ** d,))
    #Generate inout points
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
@jax.jit
def get_fpoint(x,data):
    """
    Get empirical frequencies conditioned on a point
    -------

    Parameters
    ----------
    x : jax numpy array

        Vector to conditioned on

    data : jax numpy array

        Data matrix

    Returns
    -------

    a JAX numpy array

    """
    zero = (data == jnp.append(x,0)).all(-1).sum()
    one = (data == jnp.append(x,1)).all(-1).sum()
    total = zero + one
    return jnp.append(zero,one)

#Get frequence table
def get_ftable(data,unique):
    """
    Get empirical frequencies of data matrix
    -------

    Parameters
    ----------
    data : jax numpy array

        Data matrix

    unique : logical

        If data is unique

    Returns
    -------

    a JAX numpy array

    """
    #Get dimension
    if not unique:
        domain = jnp.unique(data[:,:-1],axis = 0)
    else:
        domain = data[:,:-1]
    f = jax.vmap(lambda x: get_fpoint(x,data))(domain)
    return jnp.append(domain,f,1)
