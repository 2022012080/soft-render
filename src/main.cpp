#include "window.h"
#include <iostream>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    std::cout << "软光栅渲染器启动中..." << std::endl;
    
    RenderWindow window(820, 600);
    
    if (!window.Initialize()) {
        std::cerr << "窗口初始化失败!" << std::endl;
        return -1;
    }
    
    std::cout << "窗口创建成功，开始GUI模式..." << std::endl;
    window.Run();
    
    return 0;
}

// 为了兼容性，保留原来的main函数作为备用
int main() {
    return WinMain(GetModuleHandle(nullptr), nullptr, GetCommandLineA(), SW_SHOWNORMAL);
} 