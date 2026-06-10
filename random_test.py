import jax
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=6' # Use 6 CPU devices
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
# Mesh along one axis
mesh = jax.make_mesh((6,), ('ax1',))

def f(x):
    return x

a = jnp.arange(12)
res = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P('ax1'))(a)
print(res)
# works like np.concat([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
# res is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

res2 = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P(), check_vma=False)(a)
print(res2)
# res2 is [0, 1]
# Because shard_map has been told that f(block 0) and f(block 1/2/3) should be the same, it just returns f(block 0).
# It would normally throw an error to warn you about losing the information, (which is good), but we can disable it by setting check_vma=False.


a = jnp.arange(12).reshape(12, 1)

mesh = jax.make_mesh((3,2), ('ax1', 'ax2'))

res3 = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P('ax1', 'ax2'))(a)
print(res3)
# res3 is [[ 0,  0], [ 1,  1], [ 2,  2], [ 3,  3], [ 4,  4], [ 5,  5], [ 6,  6], [ 7,  7], [ 8,  8], [ 9,  9], [10, 10], [11, 11]]
# f(block 0) is [0, 1, 2, 3]. f(block 1) is [4, 5, 6, 7]. f(block 2) is [8, 9, 10, 11].
# They are concatenated along the 1st axis and then tiled along the 2nd axis.