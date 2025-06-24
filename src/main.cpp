#include "window.h"
#include <iostream>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    std::cout << "Soft Renderer starting..." << std::endl;
    
    RenderWindow window(1800, 1440);
    
    if (!window.Initialize()) {
        std::cerr << "Window initialization failed!" << std::endl;
        return -1;
    }
    
    std::cout << "Window created successfully, starting GUI mode..." << std::endl;
    window.Run();
    
    return 0;
}

// 为了兼容性，保留原来的main函数作为备用
int main() {
    return WinMain(GetModuleHandle(nullptr), nullptr, GetCommandLineA(), SW_SHOWNORMAL);
} 