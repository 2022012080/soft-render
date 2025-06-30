#pragma once
#include "renderer.h"

#ifdef USE_CUDA
// GPU 版渲染器：接口尽量与 Renderer 保持一致，暂仅实现 renderModel 框架
class RendererGPU : public Renderer {
public:
    using Renderer::Renderer; // 复用基类构造函数

    // 渲染模型（GPU 路径）
    void renderModel(const Model& model);
};
#endif // USE_CUDA 