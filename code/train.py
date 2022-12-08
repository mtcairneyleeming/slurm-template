## Based heavily off https://github.com/elizavetasemenova/PriorVAE/blob/main/1d_gp.ipynb, basically by cut and pasting

# general libraries
import time
import math
import pickle


# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit

# in adjacent file
from code import stax


# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, Predictive





##############
# GP Priors

def dist_euclid(x, z):
    x = jnp.array(x) 
    z = jnp.array(z)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1)
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        x_d = x[:,d]
        z_d = z[:,d]
        delta += (x_d[:,jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(x, z, var, length, noise=0, jitter=1.0e-6):
    dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k


kernels = { "exp_sq_kernel": exp_sq_kernel}


def GP(gp_kernel, x, jitter=1e-5, var=None, length=None, y=None, noise=False):
    
    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4,1))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
        
    k = gp_kernel(x, x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)
        
        
        
        
        
        
#################
# VAE defs and training


def vae_encoder(hidden_dim1, hidden_dim2, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Relu,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()), # mean
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
        ),
    )


def vae_decoder(hidden_dim1, hidden_dim2, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(out_dim, W_init=stax.randn()) 
    )


def vae_model(batch, hidden_dim1, hidden_dim2, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", vae_decoder(hidden_dim1, hidden_dim2, out_dim), (batch_dim, z_dim))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
    gen_loc = decode(z)    
    return numpyro.sample("obs", dist.Normal(gen_loc, .1), obs=batch) 
    

def vae_guide(batch, hidden_dim1, hidden_dim2, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", vae_encoder(hidden_dim1, hidden_dim2, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    z = numpyro.sample("z", dist.Normal(z_loc, z_std))
    return z

    

def epoch_train(gp, svi, args, rng_key, svi_state, num_train):

    
    @jit # moved to avoid JAX complaining about not being able to JIT a "Predictive" object
    def body_fn(i, val):
        rng_key_i = random.fold_in(rng_key, i) 
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        loss_sum, svi_state = val # val -- svi_state
        batch = gp(rng_key_i, gp_kernel=kernels[args["gp_kernel"]], x=args["x"], jitter=1e-4)
        svi_state, loss = svi.update(svi_state, batch['y']) 
        loss_sum += loss / args['batch_size']
        return loss_sum, svi_state

    return lax.fori_loop(0, num_train, body_fn, (0.0, svi_state)) #fori_loop(lower, upper, body_fun, init_val)


def eval_test(gp, svi, args, rng_key, svi_state, num_test):

    @jit
    def body_fn(i, loss_sum):
        rng_key_i = random.fold_in(rng_key, i) 
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        batch = gp(rng_key_i, gp_kernel=kernels[args["gp_kernel"]], x=args["x"], jitter=1e-4)
        loss = svi.evaluate(svi_state, batch['y']) / args['batch_size']
        loss_sum += loss
        return loss_sum

    loss = lax.fori_loop(0, num_test, body_fn, 0.0)
    loss = loss / num_test

    return loss













######### actually run it, if asked to:

def train(args):
#if __name__ == "__main__":
    print(jax.devices())
    print(jax.default_backend())
    
    
    gp_predictive = Predictive(GP, num_samples=args["batch_size"])
    ### 

    adam = optim.Adam(step_size=args["learning_rate"])

    svi = SVI(vae_model, vae_guide, adam, Trace_ELBO(), 
            hidden_dim1=args["hidden_dim1"], 
            hidden_dim2=args["hidden_dim2"], 
            z_dim=args["z_dim"])


    rng_key, rng_key_predict = random.split(random.PRNGKey(4))
    rng_key, rng_key_samp, rng_key_init = random.split(args["rng_key"], 3)
    init_batch = gp_predictive(rng_key_predict, x=args["x"], gp_kernel = kernels[args["gp_kernel"]])['y']
    svi_state = svi.init(rng_key_init, init_batch)

    test_loss_list = []

    print("Beginning training")

    for i in range(args['num_epochs']):
        
        rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
        
        t_start = time.time()

        num_train = 1000
        _, svi_state = epoch_train(gp_predictive, svi, args, rng_key_train, svi_state, num_train)

        num_test = 1000
        test_loss = eval_test(gp_predictive,svi, args, rng_key_test, svi_state, num_test)
        test_loss_list += [test_loss]

        print(
            "Epoch {}: loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            )
        )
        
        if math.isnan(test_loss): break
        
        
    ### save output

    with open('output/test_loss_list', 'wb+') as file:
        pickle.dump(test_loss_list, file)
    # Save after training

    decoder_params = svi.get_params(svi_state)["decoder$params"]
    args["decoder_params"] = decoder_params

    with open('output/decoder_1d_n400', 'wb+') as file:
        pickle.dump(decoder_params, file)
