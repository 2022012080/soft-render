#ifndef USE_CUDA
extern "C" void launchRasterizeKernel(const void*, const void*, int, int, int, void*, void*) {}
#endif 