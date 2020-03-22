
# CudaMath

C++ hardware-accelerated math engine using **Nvidia Cuda**.

## main

Runs tests on the engine and prints profiling data.  Supports redirection.

## example call

Desktop\CudaMath> cudamath 10 7 8 > profile.txt

### console

>Warming up...
>
>vectorAdd:   running...
>
>vectorInAdd: running...
>
>vectorSub:   running...
>
>vectorInSub: running...

### profile.txt

>Config: 10 tests, 10^7 vector size, 8 warps per block.
>
>vectorAdd:   27ms avg.
>
>vectorInAdd: 24ms avg.
>
>vectorSub:   27ms avg.
>
>vectorInSub: 24ms avg.

## vectorAdd

Adds 2 vectors into a third.

*Slower than cpu in most cases due to memcpy overhead.*

## vectorInAdd

Adds the second vector to the first.

*Significantly faster than **vectorAdd** on large vectors.*

## vectorSub

Subtracts 2 vectors into a third.

*Slower than cpu in most cases due to memcpy overhead.*

## vectorInSub

Subtracts the second vector from the first.

*Significantly faster than **vectorSub** on large vectors.*
