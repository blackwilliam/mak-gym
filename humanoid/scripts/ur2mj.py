
from urdf2mjcf import run


run(
    urdf_path="/home/sean-wang/wangxc/mak-gym/resources/robots/makZero/mak-zero.urdf",
    mjcf_path="/home/sean-wang/wangxc/mak-gym/resources/robots/makZero/mak-zero.mjcf",
    copy_meshes=True,
)