
# CudaMath

C++ hardware-accelerated math engine using **Nvidia Cuda**.

## vectorAdd

Adds 2 vectors into a third.

*Slower than cpu in most cases due to memcpy overhead.*

## vectorInAdd

Adds the second vector to the first.

*Significantly faster than **vectorAdd**.*
