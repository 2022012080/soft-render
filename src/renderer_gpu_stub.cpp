#ifndef USE_CUDA
#include "renderer.h"
class RendererGPU : public Renderer {
public:
    using Renderer::Renderer;
    void renderModel(const Model& model) {
        // 非 CUDA 环境：退回基类实现
        Renderer::renderModel(model);
    }
};
#endif 