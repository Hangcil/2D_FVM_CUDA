# 2D_FVM_CUDA
2D FVM solvers for the Euler equations parallelized by CUDA. This lib implements my own algorithm proposed in my academic paper. It's fast and reliable, which has implemented our algorithm faithfully and correctly. The following test is conducted on RTX 4060, finishing in 15 minutes for a 2000×2000 mesh size.

Please note that the program treats the two outer layers of cells as ghost cells by default, which generally does not affect the integrity and validity of the numerical solution provided the mesh is fine. But if you do mind please manually add two outer layers as ghost cell layers to the initial data matrix, change the spatial size accordingly, and trim the result when the computation finishes.
![](https://github.com/Hangcil/2D_FVM_CUDA/blob/main/Screenshot%202025-04-17%20202634.png)

