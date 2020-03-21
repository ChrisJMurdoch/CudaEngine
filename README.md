# CudaMath
C++ hardware-accelerated math engine using **Nvidia Cuda**.
### gpu_add
Proof of concept function to add the contents of 2 arrays into a 3rd.

*Slower than **cpu_add** in cases where size < 10^9 due to memcpy overhead.*
