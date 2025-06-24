#include "window.h"
#include "vector_math.h"
#include <commctrl.h>
#include <iostream>
#include <sstream>
#include <cmath>

RenderWindow::RenderWindow(int width, int height) 
    : m_windowWidth(width), m_windowHeight(height)
    , m_renderWidth(1200), m_renderHeight(900)
    , m_cameraX(0.0f), m_cameraY(0.0f), m_cameraZ(5.0f)
    , m_rotationX(0.0f), m_rotationY(30.0f), m_rotationZ(0.0f)
    , m_cameraRollX(0.0f), m_cameraRollY(0.0f), m_cameraRollZ(0.0f)
    , m_fov(20.0f)  // 初始FOV为20度
    , m_lightX(3.0f), m_lightY(3.0f), m_lightZ(3.0f), m_lightIntensity(10.0f)
    , m_light2X(-3.0f), m_light2Y(2.0f), m_light2Z(1.0f), m_light2Intensity(5.0f) // 第二个光源
    , m_hwnd(nullptr), m_renderArea(nullptr)
    , m_bitmap(nullptr), m_memDC(nullptr), m_bitmapData(nullptr)
{
    m_renderer = std::make_unique<Renderer>(m_renderWidth, m_renderHeight);
    m_model = std::make_unique<Model>();
    m_texture = std::make_shared<Texture>();
}

RenderWindow::~RenderWindow() {
    if (m_bitmap) {
        DeleteObject(m_bitmap);
    }
    if (m_memDC) {
        DeleteDC(m_memDC);
    }
}

bool RenderWindow::Initialize() {
    // Register window class
    WNDCLASSA wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "SoftRendererWindow";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    
    if (!RegisterClassA(&wc)) {
        return false;
    }
    
    // Create main window
    m_hwnd = CreateWindowExA(
        0,
        "SoftRendererWindow",
        "Soft Renderer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        m_windowWidth, m_windowHeight,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        this
    );
    
    if (!m_hwnd) {
        return false;
    }
    
    CreateControls();
    
    // Load model and texture
    if (m_model->loadFromFile("assets/cube.obj")) {
        m_model->centerModel();
        m_model->scaleModel(0.5f);
    }
    
    // 尝试加载BMP纹理文件，如果失败则创建默认纹理
    if (!m_texture->loadFromFile("assets/texture.bmp")) {
        std::cout << "未找到 assets/texture.bmp，尝试加载其他纹理文件..." << std::endl;
        
        // 尝试其他常见的纹理文件名
        if (!m_texture->loadFromFile("assets/cube_texture.bmp") &&
            !m_texture->loadFromFile("assets/diffuse.bmp") &&
            !m_texture->loadFromFile("texture.bmp")) {
            std::cout << "未找到纹理文件，使用默认纹理" << std::endl;
            m_texture->createDefault(256, 256);
        } else {
            std::cout << "成功加载纹理文件" << std::endl;
        }
    } else {
        std::cout << "成功加载 assets/texture.bmp" << std::endl;
    }
    
    m_renderer->setTexture(m_texture);
    
    // Create DIB for rendering
    HDC hdc = GetDC(m_hwnd);
    m_memDC = CreateCompatibleDC(hdc);
    
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = m_renderWidth;
    bmi.bmiHeader.biHeight = -m_renderHeight;  // Top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    
    m_bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &m_bitmapData, nullptr, 0);
    SelectObject(m_memDC, m_bitmap);
    
    ReleaseDC(m_hwnd, hdc);
    
    // Initial render
    UpdateRender();
    
    ShowWindow(m_hwnd, SW_SHOW);
    UpdateWindow(m_hwnd);
    
    return true;
}

void RenderWindow::CreateControls() {
    // Camera controls - 增加标签宽度和控件间距
    CreateWindowA("STATIC", "Camera Position (X, Y, Z):", WS_VISIBLE | WS_CHILD,
        1220, 20, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_cameraXEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 45, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_X, GetModuleHandle(nullptr), nullptr);
    
    m_cameraYEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1290, 45, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_Y, GetModuleHandle(nullptr), nullptr);
    
    m_cameraZEdit = CreateWindowA("EDIT", "5", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1360, 45, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_Z, GetModuleHandle(nullptr), nullptr);
    
    // Rotation controls - 增加间距
    CreateWindowA("STATIC", "Model Rotation (X, Y, Z degrees):", WS_VISIBLE | WS_CHILD,
        1220, 80, 250, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_rotationXEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 105, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_X, GetModuleHandle(nullptr), nullptr);
    
    m_rotationYEdit = CreateWindowA("EDIT", "30", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1290, 105, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_Y, GetModuleHandle(nullptr), nullptr);
    
    m_rotationZEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1360, 105, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_Z, GetModuleHandle(nullptr), nullptr);
    
    // Camera roll control - XYZ三个轴的旋转
    CreateWindowA("STATIC", "Camera Roll (X, Y, Z degrees):", WS_VISIBLE | WS_CHILD,
        1220, 140, 250, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollXEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 165, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_X, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollYEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1290, 165, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_Y, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollZEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1360, 165, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light controls - 增加间距
    CreateWindowA("STATIC", "Light Position (X, Y, Z):", WS_VISIBLE | WS_CHILD,
        1220, 200, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightXEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 225, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_X, GetModuleHandle(nullptr), nullptr);
    
    m_lightYEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1290, 225, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Y, GetModuleHandle(nullptr), nullptr);
    
    m_lightZEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1360, 225, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light intensity control
    CreateWindowA("STATIC", "Light Intensity:", WS_VISIBLE | WS_CHILD,
        1220, 260, 120, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightIntensityEdit = CreateWindowA("EDIT", "10", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 285, 80, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
    // 新增：第二个光源控制
    m_light2Label = CreateWindowA("STATIC", "Light 2 Position (X, Y, Z):", WS_VISIBLE | WS_CHILD,
        1220, 320, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_light2XEdit = CreateWindowA("EDIT", "-3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 345, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_X, GetModuleHandle(nullptr), nullptr);
    
    m_light2YEdit = CreateWindowA("EDIT", "2", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1290, 345, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_Y, GetModuleHandle(nullptr), nullptr);
    
    m_light2ZEdit = CreateWindowA("EDIT", "1", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1360, 345, 60, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light 2 intensity control
    CreateWindowA("STATIC", "Light 2 Intensity:", WS_VISIBLE | WS_CHILD,
        1220, 380, 120, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_light2IntensityEdit = CreateWindowA("EDIT", "5", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 405, 80, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
    // FOV controls
    CreateWindowA("STATIC", "Field of View (FOV):", WS_VISIBLE | WS_CHILD,
        1220, 440, 150, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_fovDecreaseBtn = CreateWindowA("BUTTON", "(-)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 490, 80, 30, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_DECREASE, GetModuleHandle(nullptr), nullptr);
    
    m_fovIncreaseBtn = CreateWindowA("BUTTON", "(+)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1310, 490, 80, 30, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_INCREASE, GetModuleHandle(nullptr), nullptr);
    
    // 新增：绘制控制按钮
    CreateWindowA("STATIC", "Render Controls:", WS_VISIBLE | WS_CHILD,
        1220, 540, 150, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleEdgesBtn = CreateWindowA("BUTTON", "Edges: ON", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 570, 100, 30, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_EDGES, GetModuleHandle(nullptr), nullptr);
    
    m_toggleRaysBtn = CreateWindowA("BUTTON", "Rays: ON", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1330, 570, 100, 30, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_RAYS, GetModuleHandle(nullptr), nullptr);
    
    // 新增：SSAA控制
    CreateWindowA("STATIC", "SSAA (Super Sampling):", WS_VISIBLE | WS_CHILD,
        1220, 610, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleSSAABtn = CreateWindowA("BUTTON", "SSAA: OFF", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 635, 100, 30, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_SSAA, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleDecBtn = CreateWindowA("BUTTON", "Scale -", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1330, 635, 70, 30, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_DEC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleIncBtn = CreateWindowA("BUTTON", "Scale +", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1410, 635, 70, 30, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_INC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaStatusLabel = CreateWindowA("STATIC", "SSAA: OFF (1x)", WS_VISIBLE | WS_CHILD,
        1220, 675, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Render area
    m_renderArea = CreateWindowA("STATIC", "", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_BITMAP,
        10, 10, m_renderWidth, m_renderHeight, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
}

void RenderWindow::Run() {
    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

LRESULT CALLBACK RenderWindow::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    RenderWindow* pThis = nullptr;
    
    if (uMsg == WM_NCCREATE) {
        CREATESTRUCT* pCreate = (CREATESTRUCT*)lParam;
        pThis = (RenderWindow*)pCreate->lpCreateParams;
        SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)pThis);
        pThis->m_hwnd = hwnd;
    } else {
        pThis = (RenderWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    }
    
    if (pThis) {
        return pThis->HandleMessage(uMsg, wParam, lParam);
    } else {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

LRESULT RenderWindow::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
        
    case WM_COMMAND:
        if (HIWORD(wParam) == EN_CHANGE) {
            int controlId = LOWORD(wParam);
            if (controlId >= ID_CAMERA_X && controlId <= ID_CAMERA_Z) {
                OnCameraChanged();
            } else if (controlId >= ID_ROTATION_X && controlId <= ID_ROTATION_Y || controlId == ID_ROTATION_Z) {
                OnRotationChanged();
            } else if (controlId >= ID_CAMERA_ROLL_X && controlId <= ID_CAMERA_ROLL_Z) {
                OnCameraChanged();  // 摄像机roll角度变化也调用OnCameraChanged
            } else if (controlId >= ID_LIGHT_X && controlId <= ID_LIGHT_INTENSITY) {
                OnLightChanged();
            } else if (controlId >= ID_LIGHT2_X && controlId <= ID_LIGHT2_INTENSITY) {
                OnLight2Changed();
            }
        } else if (HIWORD(wParam) == BN_CLICKED) {
            int controlId = LOWORD(wParam);
            if (controlId == ID_FOV_INCREASE) {
                OnFovIncrease();
            } else if (controlId == ID_FOV_DECREASE) {
                OnFovDecrease();
            } else if (controlId == ID_TOGGLE_EDGES) {
                OnToggleEdges();
            } else if (controlId == ID_TOGGLE_RAYS) {
                OnToggleRays();
            } else if (controlId == ID_TOGGLE_SSAA) {
                OnToggleSSAA();
            } else if (controlId == ID_SSAA_SCALE_INC) {
                OnSSAAScaleIncrease();
            } else if (controlId == ID_SSAA_SCALE_DEC) {
                OnSSAAScaleDecrease();
            }
        }
        return 0;
        
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(m_hwnd, &ps);
        
        // Get the render area position
        RECT renderRect;
        GetWindowRect(m_renderArea, &renderRect);
        POINT pt = {renderRect.left, renderRect.top};
        ScreenToClient(m_hwnd, &pt);
        
        // Blit the rendered image directly to the main window at render area position
        BitBlt(hdc, pt.x, pt.y, m_renderWidth, m_renderHeight,
               m_memDC, 0, 0, SRCCOPY);
        
        EndPaint(m_hwnd, &ps);
        return 0;
    }
    
    default:
        return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
    }
}

void RenderWindow::OnCameraChanged() {
    // Get camera values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_cameraXEdit, buffer, sizeof(buffer));
    m_cameraX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraYEdit, buffer, sizeof(buffer));
    m_cameraY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraZEdit, buffer, sizeof(buffer));
    m_cameraZ = static_cast<float>(atof(buffer));
    
    // Get camera roll angles for XYZ axes
    GetWindowTextA(m_cameraRollXEdit, buffer, sizeof(buffer));
    m_cameraRollX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraRollYEdit, buffer, sizeof(buffer));
    m_cameraRollY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraRollZEdit, buffer, sizeof(buffer));
    m_cameraRollZ = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::OnRotationChanged() {
    // Get rotation values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_rotationXEdit, buffer, sizeof(buffer));
    m_rotationX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_rotationYEdit, buffer, sizeof(buffer));
    m_rotationY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_rotationZEdit, buffer, sizeof(buffer));
    m_rotationZ = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::OnLightChanged() {
    // Get light values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_lightXEdit, buffer, sizeof(buffer));
    m_lightX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_lightYEdit, buffer, sizeof(buffer));
    m_lightY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_lightZEdit, buffer, sizeof(buffer));
    m_lightZ = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_lightIntensityEdit, buffer, sizeof(buffer));
    m_lightIntensity = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::OnLight2Changed() {
    // Get light values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_light2XEdit, buffer, sizeof(buffer));
    m_light2X = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_light2YEdit, buffer, sizeof(buffer));
    m_light2Y = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_light2ZEdit, buffer, sizeof(buffer));
    m_light2Z = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_light2IntensityEdit, buffer, sizeof(buffer));
    m_light2Intensity = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::UpdateRender() {
    // Set up matrices - 支持完整的XYZ旋转
    Matrix4x4 modelMatrix = VectorMath::translate(Vec3f(0, 0, -5)) * 
                           VectorMath::rotate(m_rotationZ, Vec3f(0, 0, 1)) *  // Z轴旋转
                           VectorMath::rotate(m_rotationY, Vec3f(0, 1, 0)) *  // Y轴旋转
                           VectorMath::rotate(m_rotationX, Vec3f(1, 0, 0)) *  // X轴旋转
                           VectorMath::scale(Vec3f(0.5f, 0.5f, 0.5f));
    
    // 计算带有XYZ三轴旋转的摄像机变换
    Vec3f cameraPos(m_cameraX, m_cameraY, m_cameraZ);
    Vec3f target(0, 0, 0);
    
    // 基础的LookAt矩阵
    Vec3f forward = (target - cameraPos).normalize();
    Vec3f worldUp(0, 1, 0);
    Vec3f right = forward.cross(worldUp).normalize();
    Vec3f up = right.cross(forward).normalize();
    
    Matrix4x4 baseLookAt = VectorMath::lookAt(cameraPos, target, up);
    
    // 应用摄像机的XYZ轴旋转
    Matrix4x4 cameraRotation = VectorMath::rotate(m_cameraRollZ, Vec3f(0, 0, 1)) *  // Z轴旋转 (Roll)
                              VectorMath::rotate(m_cameraRollY, Vec3f(0, 1, 0)) *  // Y轴旋转 (Yaw)
                              VectorMath::rotate(m_cameraRollX, Vec3f(1, 0, 0));   // X轴旋转 (Pitch)
    
    Matrix4x4 viewMatrix = baseLookAt * cameraRotation;
    
    Matrix4x4 projectionMatrix = VectorMath::perspective(m_fov, 
        (float)m_renderWidth / m_renderHeight, 0.1f, 100.0f);
    
    Matrix4x4 viewportMatrix;
    viewportMatrix(0, 0) = m_renderWidth / 2.0f;
    viewportMatrix(1, 1) = m_renderHeight / 2.0f;
    viewportMatrix(2, 2) = 1.0f;
    viewportMatrix(0, 3) = m_renderWidth / 2.0f;
    viewportMatrix(1, 3) = m_renderHeight / 2.0f;
    
    m_renderer->setModelMatrix(modelMatrix);
    m_renderer->setViewMatrix(viewMatrix);
    m_renderer->setProjectionMatrix(projectionMatrix);
    m_renderer->setViewportMatrix(viewportMatrix);
    
    // 更新多光源系统
    if (m_renderer->getLightCount() >= 2) {
        // 更新第一个光源
        Light& light1 = m_renderer->getLight(0);
        light1.position = Vec3f(m_lightX, m_lightY, m_lightZ);
        light1.color = Vec3f(1, 1, 1); // 白色
        light1.intensity = m_lightIntensity;
        
        // 更新第二个光源
        Light& light2 = m_renderer->getLight(1);
        light2.position = Vec3f(m_light2X, m_light2Y, m_light2Z);
        light2.color = Vec3f(0.8, 0.6, 1.0); // 紫色
        light2.intensity = m_light2Intensity;
    }
    
    m_renderer->setAmbientIntensity(0.3f);
    
    // Clear and render
    m_renderer->clear(Color(50, 50, 100));
    m_renderer->clearDepth();
    
    if (m_model->getFaceCount() > 0) {
        // 渲染模型（包括三角形和边界线）
        m_renderer->renderModel(*m_model);
    }
    
    // 在模型渲染后绘制坐标轴、网格、光源位置和光线
    // 这样它们不会被SSAA的下采样过程覆盖
    m_renderer->drawGrid(5.0f, 5);   // 5单位大小，5个分割（每1单位一条线）
    m_renderer->drawAxes(2.0f);      // 2单位长度的坐标轴
    m_renderer->drawAllLightPositions(); // 绘制所有光源位置
    
    if (m_model->getFaceCount() > 0) {
        // 根据开关决定是否绘制光线
        if (m_renderer->getDrawLightRays()) {
            m_renderer->drawLightRays(*m_model); // 绘制光线到顶点
        }
    }
    
    RenderToWindow();
}

void RenderWindow::RenderToWindow() {
    // Copy renderer's color buffer to the DIB
    const auto& colorBuffer = m_renderer->getColorBuffer();
    
    if (m_bitmapData && !colorBuffer.empty()) {
        DWORD* pixels = (DWORD*)m_bitmapData;
        
        for (int y = 0; y < m_renderHeight; y++) {
            for (int x = 0; x < m_renderWidth; x++) {
                int srcIndex = y * m_renderWidth + x;
                if (srcIndex < colorBuffer.size()) {
                    const Color& color = colorBuffer[srcIndex];
                    // For 32-bit DIB, the format is 0x00BBGGRR (BGR format)
                    pixels[srcIndex] = (0xFF << 24) | (color.r << 16) | (color.g << 8) | color.b;
                }
            }
        }
    }
    
    // Trigger repaint of the main window to refresh the render area
    InvalidateRect(m_hwnd, nullptr, FALSE);
    UpdateWindow(m_hwnd);
}

// FOV控制方法实现
void RenderWindow::OnFovIncrease() {
    m_fov += 5.0f;  // 每次增加5度
    if (m_fov > 120.0f) {  // 限制最大FOV为120度
        m_fov = 120.0f;
    }
    UpdateRender();
}

void RenderWindow::OnFovDecrease() {
    m_fov -= 5.0f;  // 每次减少5度
    if (m_fov < 10.0f) {  // 限制最小FOV为10度
        m_fov = 10.0f;
    }
    UpdateRender();
}

void RenderWindow::OnToggleEdges() {
    bool currentState = m_renderer->getDrawTriangleEdges();
    m_renderer->setDrawTriangleEdges(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Edges: ON" : "Edges: OFF";
    SetWindowTextA(m_toggleEdgesBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleRays() {
    bool currentState = m_renderer->getDrawLightRays();
    m_renderer->setDrawLightRays(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Rays: ON" : "Rays: OFF";
    SetWindowTextA(m_toggleRaysBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleSSAA() {
    bool currentState = m_renderer->isSSAAEnabled();
    
    if (currentState) {
        m_renderer->disableSSAA();
        SetWindowTextA(m_toggleSSAABtn, "SSAA: OFF");
    } else {
        m_renderer->enableSSAA(true, m_renderer->getSSAAScale());
        SetWindowTextA(m_toggleSSAABtn, "SSAA: ON");
    }
    
    UpdateSSAAControls();
    UpdateRender();
}

void RenderWindow::OnSSAAScaleIncrease() {
    int currentScale = m_renderer->getSSAAScale();
    if (currentScale < 8) {  // 限制最大8x
        int newScale = currentScale * 2;  // 2x, 4x, 8x
        bool wasEnabled = m_renderer->isSSAAEnabled();
        
        if (wasEnabled) {
            m_renderer->enableSSAA(true, newScale);
        } else {
            // 即使未启用，也更新比例设置
            m_renderer->enableSSAA(false, newScale);
        }
        
        UpdateSSAAControls();
        if (wasEnabled) {
            UpdateRender();
        }
    }
}

void RenderWindow::OnSSAAScaleDecrease() {
    int currentScale = m_renderer->getSSAAScale();
    if (currentScale > 2) {  // 限制最小2x
        int newScale = currentScale / 2;  // 8x, 4x, 2x
        bool wasEnabled = m_renderer->isSSAAEnabled();
        
        if (wasEnabled) {
            m_renderer->enableSSAA(true, newScale);
        } else {
            // 即使未启用，也更新比例设置
            m_renderer->enableSSAA(false, newScale);
        }
        
        UpdateSSAAControls();
        if (wasEnabled) {
            UpdateRender();
        }
    }
}

void RenderWindow::UpdateSSAAControls() {
    bool isEnabled = m_renderer->isSSAAEnabled();
    int scale = m_renderer->getSSAAScale();
    
    // 更新状态标签
    char statusText[64];
    if (isEnabled) {
        int highResWidth = m_renderWidth * scale;
        int highResHeight = m_renderHeight * scale;
        sprintf_s(statusText, sizeof(statusText), "SSAA: ON (%dx) [%dx%d->%dx%d]", 
                 scale, highResWidth, highResHeight, m_renderWidth, m_renderHeight);
    } else {
        sprintf_s(statusText, sizeof(statusText), "SSAA: OFF (%dx available)", scale);
    }
    
    SetWindowTextA(m_ssaaStatusLabel, statusText);
    
    // 更新按钮状态
    EnableWindow(m_ssaaScaleIncBtn, scale < 8);
    EnableWindow(m_ssaaScaleDecBtn, scale > 2);
} 