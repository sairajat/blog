---
title: 🔍 Know thy GPU - A Fun Dive into CUDA Device Introspection
date: 2025-06-10 09:15:38
tags:
---
 

Ever wondered what your GPU is made of? I don’t mean physically (though that would make a great teardown video) — I mean *capability-wise*. If you’re working with CUDA, it’s crucial to know whether your GPU supports managed memory, tensor cores, or concurrent kernel execution. And hey, maybe you're just trying to settle a bet about whose card is faster. 🏎️

In this post, we'll go on a quick and entertaining tour through a powerful C++ tool that queries all your CUDA-capable GPUs and tells you everything from warp size to peak memory bandwidth. Buckle up!

<!-- more -->
---

## 🧭 What We'll Do

We'll walk through a simple (but mighty!) C++ program that:

* Detects all CUDA GPUs on your machine
* Prints detailed properties like compute capability, memory specs, and core clock
* Tells you if your GPU can juggle multiple tasks like a caffeinated octopus 🐙

All this, using `cudaDeviceProp` and a sprinkle of `std::cout` magic.

---

## 💻 The Full Code

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "====================================\n";
        std::cout << "Device Number: " << i " of " << nDevices << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;

        std::cout << "  Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "  Tensor Cores Available: " << (prop.major >= 7 ? "yes" : "no") << std::endl;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Clock Rate (Core MHz): " << prop.clockRate / 1000.0 << std::endl;
        std::cout << "  Multiprocessor Clock Rate (MHz): " << prop.clockRate / 1000.0 << std::endl;

        std::cout << std::setprecision(0);
        std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate / 1024 << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;

        std::cout << std::fixed << std::setprecision(1);
        double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        std::cout << "  Peak Memory Bandwidth (GB/s): " << bandwidth << std::endl;

        std::cout << "  Total global memory (Gbytes): "
                  << static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024) << std::endl;
        std::cout << "  Total constant memory (Kbytes): "
                  << static_cast<float>(prop.totalConstMem) / 1024.0 << std::endl;
        std::cout << "  Shared memory per block (Kbytes): "
                  << static_cast<float>(prop.sharedMemPerBlock) / 1024.0 << std::endl;
        std::cout << "  L2 Cache Size (Kbytes): "
                  << static_cast<float>(prop.l2CacheSize) / 1024.0 << std::endl;

        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

        std::cout << "  Max threads dim: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";

        std::cout << "  Max grid size: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";

        std::cout << "  Async Engines: " << prop.asyncEngineCount << std::endl;
        std::cout << "  Concurrent Kernels: "
                  << (prop.concurrentKernels ? "yes" : "no") << std::endl;
        std::cout << "  Concurrent Copy and Execution: "
                  << (prop.deviceOverlap ? "yes" : "no") << std::endl;

        std::cout << "  Unified Addressing: "
                  << (prop.unifiedAddressing ? "yes" : "no") << std::endl;
        std::cout << "  Managed Memory Supported: "
                  << (prop.managedMemory ? "yes" : "no") << std::endl;

        std::cout << "  PCI Bus ID / Device ID: "
                  << prop.pciBusID << " / " << prop.pciDeviceID << std::endl;
        std::cout << "====================================\n" << std::endl;
    }

    return 0;
}
```

---

## 🔎 Why This Matters

Understanding your GPU's hardware capabilities is like knowing your car's horsepower before entering a drag race. It tells you:

* Whether you can use advanced CUDA features like Unified Memory or Tensor Cores
* How much parallelism you can exploit (SMs, warps, threads)
* If your hardware is limiting your algorithm’s performance (e.g., low memory bandwidth)
* What optimization knobs you can safely ignore or push harder

It’s not just about geeking out (although that’s half the fun) — it's about *writing better, faster GPU code*.

---

## 🧠 Conclusion

With just a few lines of C++ and CUDA runtime API, you now have a powerful utility to peek under the hood of any GPU. This kind of introspection is essential when tuning performance or building systems that must adapt to the GPU they run on.

So next time someone says, "My GPU is faster," you can pull out this program and say, "Prove it."

Happy hacking! 🚀
