# Parallel-Image-Blurring

Program i wrote to Blur Image in Parallel, this divide image to as many parts as person selects cores than apply blur on every part in parallel, then combine that parts and make final image


# Prerequisites

- install mpi library
- your machine should support multicore

# How to Build
```mpicxx -o blur_mpi blur_mpi.cpp lodepng.cpp```

# How to Run

```mpiexec -n <number_of_cores> ./blur_mpi input.png output.png```

<br>
Example:
<br>

```mpiexec -n 4 ./blur_mpi input.png output.png```

<br>
