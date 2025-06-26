#include "window.h"
#include "vector_math.h"
#include <commctrl.h>
#include <iostream>
#include <sstream>
#include <cmath>

RenderWindow::RenderWindow(int width, int height) 
    : m_windowWidth(width), m_windowHeight(height)
    , m_renderWidth(1200), m_renderHeight(900)
    , m_rotationX(0.0f), m_rotationY(0.0f), m_rotationZ(0.0f)
    , m_cameraRollX(0.0f), m_cameraRollY(0.0f), m_cameraRollZ(0.0f)
    , m_objectX(0.0f), m_objectY(0.0f), m_objectZ(-5.0f)
    , m_fov(20.0f)  // 初始FOV为20度
    , m_lightX(3.0f), m_lightY(-2.0f), m_lightZ(3.0f), m_lightIntensity(30.0f)
    , m_light2X(-3.0f), m_light2Y(2.0f), m_light2Z(1.0f), m_light2Intensity(0.0f) // 第二个光源
    , m_diffuseStrength(0.2f), m_specularStrength(2.0f), m_ambientStrength(0.15f) // 光照系数
    , m_shininess(128.0f) // 新增：高光指数初始化
    , m_hwnd(nullptr), m_renderArea(nullptr)
    , m_bitmap(nullptr), m_memDC(nullptr), m_bitmapData(nullptr)
    , m_currentModelFile("sphere.obj") // 默认模型文件
    , m_currentTextureFile("texture.bmp") // 默认纹理文件
    , m_currentNormalMapFile("normal.bmp") // 默认法线贴图文件
    , m_roughness(0.5f)        // 新增：粗糙度初始化
    , m_metallic(0.0f)         // 新增：金属度初始化
    , m_energyCompensationScale(1.0f)  // 新增：能量补偿强度初始化
{
    m_renderer = std::make_unique<Renderer>(m_renderWidth, m_renderHeight);
    m_model = std::make_unique<Model>();
    m_texture = std::make_shared<Texture>();
    m_normalMap = std::make_shared<Texture>();
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
    std::string modelPath = "assets/" + m_currentModelFile;
    if (m_model->loadFromFile(modelPath)) {
        m_model->centerModel();
        m_model->scaleModel(0.5f);
        std::cout << "成功加载模型: " << modelPath << std::endl;
    } else {
        std::cout << "加载模型失败: " << modelPath << std::endl;
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
    
    // 尝试加载默认法线贴图
    if (m_normalMap->loadFromFile("assets/" + m_currentNormalMapFile)) {
        std::cout << "成功加载默认法线贴图: " << m_currentNormalMapFile << std::endl;
        m_renderer->setNormalMap(m_normalMap);
    } else {
        std::cout << "未找到默认法线贴图: " << m_currentNormalMapFile << std::endl;
    }
    
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
    
    // 初始化所有参数到渲染器 - 确保默认值生效
    OnObjectChanged();    // 物体位置
    OnRotationChanged();  // 模型旋转
    OnCameraChanged();    // 摄像机角度
    OnLightChanged();     // 第一个光源
    OnLight2Changed();    // 第二个光源
    OnLightingChanged();  // 光照系数
    
    // 设置能量补偿复选框的默认状态为选中
    SendMessage(m_energyCompensationCheckbox, BM_SETCHECK, BST_CHECKED, 0);
    
    // Initial render
    UpdateRender();
    
    ShowWindow(m_hwnd, SW_SHOW);
    UpdateWindow(m_hwnd);
    
    return true;
}

void RenderWindow::CreateControls() {
    // Object position controls
    CreateWindowA("STATIC", "Obj Pos (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 10, 120, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_objectXEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 25, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_OBJECT_X, GetModuleHandle(nullptr), nullptr);
    
    m_objectYEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 25, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_OBJECT_Y, GetModuleHandle(nullptr), nullptr);
    
    m_objectZEdit = CreateWindowA("EDIT", "-5.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 25, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_OBJECT_Z, GetModuleHandle(nullptr), nullptr);
    
    // Rotation controls
    CreateWindowA("STATIC", "Rotation (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 50, 120, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_rotationXEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 65, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_X, GetModuleHandle(nullptr), nullptr);
    
    m_rotationYEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 65, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_Y, GetModuleHandle(nullptr), nullptr);
    
    m_rotationZEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 65, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_Z, GetModuleHandle(nullptr), nullptr);
    
    // Camera roll control
    CreateWindowA("STATIC", "Camera (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 90, 120, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollXEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 105, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_X, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollYEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 105, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_Y, GetModuleHandle(nullptr), nullptr);
    
    m_cameraRollZEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 105, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_ROLL_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light controls
    CreateWindowA("STATIC", "Light1 (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 130, 100, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightXEdit = CreateWindowA("EDIT", "3.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 145, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_X, GetModuleHandle(nullptr), nullptr);
    
    m_lightYEdit = CreateWindowA("EDIT", "-2.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 145, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Y, GetModuleHandle(nullptr), nullptr);
    
    m_lightZEdit = CreateWindowA("EDIT", "3.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 145, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light intensity control
    CreateWindowA("STATIC", "Intensity:", WS_VISIBLE | WS_CHILD,
        1370, 130, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightIntensityEdit = CreateWindowA("EDIT", "30.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1370, 145, 50, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
    // Light 2 controls
    m_light2Label = CreateWindowA("STATIC", "Light2 (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 170, 100, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_light2XEdit = CreateWindowA("EDIT", "-3.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 185, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_X, GetModuleHandle(nullptr), nullptr);
    
    m_light2YEdit = CreateWindowA("EDIT", "2.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 185, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_Y, GetModuleHandle(nullptr), nullptr);
    
    m_light2ZEdit = CreateWindowA("EDIT", "1.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 185, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light 2 intensity control
    CreateWindowA("STATIC", "Intensity:", WS_VISIBLE | WS_CHILD,
        1370, 170, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_light2IntensityEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1370, 185, 50, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT2_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
    // Lighting coefficients
    m_lightingLabel = CreateWindowA("STATIC", "Lighting:", WS_VISIBLE | WS_CHILD,
        1220, 210, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Diff:", WS_VISIBLE | WS_CHILD,
        1220, 225, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_diffuseEdit = CreateWindowA("EDIT", "0.2", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 240, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIFFUSE_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Spec:", WS_VISIBLE | WS_CHILD,
        1270, 225, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_specularEdit = CreateWindowA("EDIT", "2.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 240, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_SPECULAR_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Amb:", WS_VISIBLE | WS_CHILD,
        1320, 225, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_ambientEdit = CreateWindowA("EDIT", "0.15", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 240, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_AMBIENT_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    // Shininess control
    CreateWindowA("STATIC", "Shin:", WS_VISIBLE | WS_CHILD,
        1370, 225, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_shininessEdit = CreateWindowA("EDIT", "128.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1370, 240, 50, 20, m_hwnd, (HMENU)(LONG_PTR)ID_SHININESS, GetModuleHandle(nullptr), nullptr);
    
    // FOV controls
    CreateWindowA("STATIC", "FOV:", WS_VISIBLE | WS_CHILD,
        1220, 265, 30, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_fovDecreaseBtn = CreateWindowA("BUTTON", "(-)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 280, 30, 25, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_DECREASE, GetModuleHandle(nullptr), nullptr);
    
    m_fovIncreaseBtn = CreateWindowA("BUTTON", "(+)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1255, 280, 30, 25, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_INCREASE, GetModuleHandle(nullptr), nullptr);
    
    // Render controls
    CreateWindowA("STATIC", "Render:", WS_VISIBLE | WS_CHILD,
        1220, 310, 50, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleEdgesBtn = CreateWindowA("BUTTON", "Edge", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 325, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_EDGES, GetModuleHandle(nullptr), nullptr);
    
    m_toggleRaysBtn = CreateWindowA("BUTTON", "Ray", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1265, 325, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_RAYS, GetModuleHandle(nullptr), nullptr);
    
    // Texture controls
    m_toggleTextureBtn = CreateWindowA("BUTTON", "Tex", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1310, 325, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_TEXTURE, GetModuleHandle(nullptr), nullptr);
    
    // Normal map control
    m_toggleNormalMapBtn = CreateWindowA("BUTTON", "Norm", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1355, 325, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_NORMAL_MAP, GetModuleHandle(nullptr), nullptr);
    
    // Axes/Grid control
    m_toggleAxesGridBtn = CreateWindowA("BUTTON", "Grid", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1400, 325, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_AXES_GRID, GetModuleHandle(nullptr), nullptr);
    
    // SSAA controls
    CreateWindowA("STATIC", "SSAA:", WS_VISIBLE | WS_CHILD,
        1220, 355, 40, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleSSAABtn = CreateWindowA("BUTTON", "OFF", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 370, 35, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_SSAA, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleDecBtn = CreateWindowA("BUTTON", "-", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1260, 370, 25, 25, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_DEC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleIncBtn = CreateWindowA("BUTTON", "+", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1290, 370, 25, 25, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_INC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaStatusLabel = CreateWindowA("STATIC", "OFF", WS_VISIBLE | WS_CHILD,
        1320, 355, 100, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Model file controls
    CreateWindowA("STATIC", "Model:", WS_VISIBLE | WS_CHILD,
        1220, 400, 50, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_modelFileEdit = CreateWindowA("EDIT", "sphere.obj", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 415, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_MODEL_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadModelBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 415, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_MODEL, GetModuleHandle(nullptr), nullptr);
    
    m_modelStatusLabel = CreateWindowA("STATIC", "sphere.obj", WS_VISIBLE | WS_CHILD,
        1390, 415, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Texture file controls
    CreateWindowA("STATIC", "Texture:", WS_VISIBLE | WS_CHILD,
        1220, 440, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_textureFileEdit = CreateWindowA("EDIT", "texture.bmp", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 455, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_TEXTURE_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadTextureBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 455, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_TEXTURE, GetModuleHandle(nullptr), nullptr);
    
    m_textureStatusLabel = CreateWindowA("STATIC", "texture.bmp", WS_VISIBLE | WS_CHILD,
        1390, 455, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Normal map file controls
    CreateWindowA("STATIC", "Normal:", WS_VISIBLE | WS_CHILD,
        1220, 480, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_normalMapFileEdit = CreateWindowA("EDIT", "normal.bmp", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 495, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_NORMAL_MAP_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadNormalMapBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 495, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_NORMAL_MAP, GetModuleHandle(nullptr), nullptr);
    
    m_normalMapStatusLabel = CreateWindowA("STATIC", "normal.bmp", WS_VISIBLE | WS_CHILD,
        1390, 495, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Render area
    m_renderArea = CreateWindowA("STATIC", "", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_BITMAP,
        10, 10, m_renderWidth, m_renderHeight, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // BRDF Model controls
    CreateWindowA("STATIC", "BRDF Model:", WS_VISIBLE | WS_CHILD,
        1220, 520, 80, 15, m_hwnd, NULL, NULL, NULL);
    
    m_brdfCheckbox = CreateWindowA("BUTTON", "Enable BRDF", 
                                   WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                   1220, 535, 100, 20, m_hwnd, 
                                   (HMENU)ID_BRDF_ENABLE, NULL, NULL);
    
    // Roughness control
    CreateWindowA("STATIC", "Rough:", WS_VISIBLE | WS_CHILD,
        1220, 560, 45, 15, m_hwnd, NULL, NULL, NULL);
    
    m_roughnessEdit = CreateWindowA("EDIT", "0.5", 
                                    WS_VISIBLE | WS_CHILD | WS_BORDER,
                                    1270, 560, 50, 20, m_hwnd, 
                                    (HMENU)ID_ROUGHNESS, NULL, NULL);
    
    // Metallic control
    CreateWindowA("STATIC", "Metal:", WS_VISIBLE | WS_CHILD,
        1330, 560, 40, 15, m_hwnd, NULL, NULL, NULL);
    
    m_metallicEdit = CreateWindowA("EDIT", "0.0", 
                                   WS_VISIBLE | WS_CHILD | WS_BORDER,
                                   1375, 560, 50, 20, m_hwnd, 
                                   (HMENU)ID_METALLIC, NULL, NULL);
    
    // 能量补偿控件
    m_energyCompensationCheckbox = CreateWindowA("BUTTON", "Energy Compensation", 
                                                  WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                                  1220, 585, 140, 20, m_hwnd, 
                                                  (HMENU)ID_ENERGY_COMPENSATION_ENABLE, NULL, NULL);
    
    CreateWindowA("STATIC", "EC Scale:", WS_VISIBLE | WS_CHILD,
        1220, 610, 60, 15, m_hwnd, NULL, NULL, NULL);
    
    m_energyCompensationScaleEdit = CreateWindowA("EDIT", "1.0", 
                                                   WS_VISIBLE | WS_CHILD | WS_BORDER,
                                                   1285, 610, 50, 20, m_hwnd, 
                                                   (HMENU)ID_ENERGY_COMPENSATION_SCALE, NULL, NULL);
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
            if (controlId >= ID_OBJECT_X && controlId <= ID_OBJECT_Z) {
                OnObjectChanged();
            } else if (controlId >= ID_ROTATION_X && controlId <= ID_ROTATION_Y || controlId == ID_ROTATION_Z) {
                OnRotationChanged();
            } else if (controlId >= ID_CAMERA_ROLL_X && controlId <= ID_CAMERA_ROLL_Z) {
                OnCameraChanged();  // 摄像机roll角度变化也调用OnCameraChanged
            } else if (controlId >= ID_LIGHT_X && controlId <= ID_LIGHT_INTENSITY) {
                OnLightChanged();
            } else if (controlId >= ID_LIGHT2_X && controlId <= ID_LIGHT2_INTENSITY) {
                OnLight2Changed();
            } else if (controlId >= ID_DIFFUSE_STRENGTH && controlId <= ID_AMBIENT_STRENGTH) {
                OnLightingChanged();
            } else if (controlId == ID_SHININESS) {
                OnLightingChanged();  // shininess变化也调用OnLightingChanged
            } else if (controlId == ID_ROUGHNESS) {
                OnBRDFParameterChanged();
            } else if (controlId == ID_METALLIC) {
                OnBRDFParameterChanged();
            } else if (controlId == ID_ENERGY_COMPENSATION_SCALE) {
                OnBRDFParameterChanged();
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
            } else if (controlId == ID_TOGGLE_TEXTURE) {
                OnToggleTexture();
            } else if (controlId == ID_TOGGLE_NORMAL_MAP) {
                OnToggleNormalMap();
            } else if (controlId == ID_TOGGLE_AXES_GRID) {
                OnToggleAxesGrid();
            } else if (controlId == ID_TOGGLE_SSAA) {
                OnToggleSSAA();
            } else if (controlId == ID_SSAA_SCALE_INC) {
                OnSSAAScaleIncrease();
            } else if (controlId == ID_SSAA_SCALE_DEC) {
                OnSSAAScaleDecrease();
            } else if (controlId == ID_LOAD_MODEL) {
                OnLoadModel();
            } else if (controlId == ID_LOAD_TEXTURE) {
                OnLoadTexture();
            } else if (controlId == ID_LOAD_NORMAL_MAP) {
                OnLoadNormalMap();
            } else if (controlId == ID_BRDF_ENABLE) {
                bool enabled = (SendMessage(m_brdfCheckbox, BM_GETCHECK, 0, 0) == BST_CHECKED);
                m_renderer->setBRDFEnabled(enabled);
                UpdateRender();
            } else if (controlId == ID_ENERGY_COMPENSATION_ENABLE) {
                bool enabled = (SendMessage(m_energyCompensationCheckbox, BM_GETCHECK, 0, 0) == BST_CHECKED);
                m_renderer->setEnergyCompensationEnabled(enabled);
                UpdateRender();
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
    // Get camera roll angles for XYZ axes
    char buffer[32];
    
    GetWindowTextA(m_cameraRollXEdit, buffer, sizeof(buffer));
    m_cameraRollX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraRollYEdit, buffer, sizeof(buffer));
    m_cameraRollY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_cameraRollZEdit, buffer, sizeof(buffer));
    m_cameraRollZ = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::OnObjectChanged() {
    // Get object position values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_objectXEdit, buffer, sizeof(buffer));
    m_objectX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_objectYEdit, buffer, sizeof(buffer));
    m_objectY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_objectZEdit, buffer, sizeof(buffer));
    m_objectZ = static_cast<float>(atof(buffer));
    
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

void RenderWindow::OnLightingChanged() {
    // Get lighting coefficient values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_diffuseEdit, buffer, sizeof(buffer));
    m_diffuseStrength = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_specularEdit, buffer, sizeof(buffer));
    m_specularStrength = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_ambientEdit, buffer, sizeof(buffer));
    m_ambientStrength = static_cast<float>(atof(buffer));
    
    // 新增：读取shininess参数
    GetWindowTextA(m_shininessEdit, buffer, sizeof(buffer));
    m_shininess = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::UpdateRender() {
    // Set up matrices - 使用物体坐标参数代替固定的(0,0,-5)
    Matrix4x4 modelMatrix = VectorMath::translate(Vec3f(m_objectX, m_objectY, m_objectZ)) * 
                           VectorMath::rotate(m_rotationZ, Vec3f(0, 0, 1)) *  // Z轴旋转
                           VectorMath::rotate(m_rotationY, Vec3f(0, 1, 0)) *  // Y轴旋转
                           VectorMath::rotate(m_rotationX, Vec3f(1, 0, 0)) *  // X轴旋转
                           VectorMath::scale(Vec3f(0.5f, 0.5f, 0.5f));
    
    // 摄像机固定在原点，朝向原点，应用摄像机角度旋转
    Vec3f cameraPos(0, 0, 0);  // 摄像机固定在原点
    Vec3f target(0, 0, -1);    // 摄像机朝向负Z方向
    Vec3f worldUp(0, 1, 0);
    
    // 创建基础视图矩阵（摄像机在原点朝向-Z）
    Matrix4x4 baseViewMatrix;
    baseViewMatrix(0, 0) = 1;  baseViewMatrix(0, 1) = 0;  baseViewMatrix(0, 2) = 0;  baseViewMatrix(0, 3) = 0;
    baseViewMatrix(1, 0) = 0;  baseViewMatrix(1, 1) = 1;  baseViewMatrix(1, 2) = 0;  baseViewMatrix(1, 3) = 0;
    baseViewMatrix(2, 0) = 0;  baseViewMatrix(2, 1) = 0;  baseViewMatrix(2, 2) = 1;  baseViewMatrix(2, 3) = 0;
    baseViewMatrix(3, 0) = 0;  baseViewMatrix(3, 1) = 0;  baseViewMatrix(3, 2) = 0;  baseViewMatrix(3, 3) = 1;
    
    // 应用摄像机的XYZ轴旋转
    Matrix4x4 cameraRotation = VectorMath::rotate(m_cameraRollZ, Vec3f(0, 0, 1)) *  // Z轴旋转 (Roll)
                              VectorMath::rotate(m_cameraRollY, Vec3f(0, 1, 0)) *  // Y轴旋转 (Yaw)
                              VectorMath::rotate(m_cameraRollX, Vec3f(1, 0, 0));   // X轴旋转 (Pitch)
    
    Matrix4x4 viewMatrix = baseViewMatrix * cameraRotation;
    
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
    
    // 设置光照系数
    m_renderer->setDiffuseStrength(m_diffuseStrength);
    m_renderer->setSpecularStrength(m_specularStrength);
    m_renderer->setAmbientStrength(m_ambientStrength);
    m_renderer->setShininess(m_shininess);  // 新增：设置高光指数
    
    // 设置 BRDF 参数
    m_renderer->setRoughness(m_roughness);
    m_renderer->setMetallic(m_metallic);
    m_renderer->setEnergyCompensationScale(m_energyCompensationScale);
    
    // 根据材质类型设置 F0
    if (m_metallic > 0.5f) {
        // 金属材质使用反照率作为 F0
        m_renderer->setF0(Vec3f(0.7f, 0.7f, 0.7f));  // 银色金属
    } else {
        // 非金属材质使用标准 F0
        m_renderer->setF0(Vec3f(0.04f, 0.04f, 0.04f));
    }
    
    // Clear and render
    m_renderer->clear(Color(50, 50, 100));
    m_renderer->clearDepth();
    
    // 在SSAA模式下，需要在模型渲染前绘制网格和坐标轴，这样它们能被正确遮挡
    if (m_renderer->isSSAAEnabled()) {
        // 先绘制背景元素（网格、坐标轴）- 根据开关决定是否绘制
        if (m_renderer->getDrawAxesAndGrid()) {
            m_renderer->drawGrid(5.0f, 5);   // 5单位大小，5个分割（每1单位一条线）
            m_renderer->drawAxes(2.0f);      // 2单位长度的坐标轴
        }
        
        // 然后渲染模型（在高分辨率下，会被正确遮挡）
        if (m_model->getFaceCount() > 0) {
            m_renderer->renderModel(*m_model);
        }
        
        // 最后绘制光源位置和光线（总是显示在最前面）
        m_renderer->drawAllLightPositions();
        if (m_model->getFaceCount() > 0) {
            if (m_renderer->getDrawLightRays()) {
                m_renderer->drawLightRays(*m_model);
            }
        }
    } else {
        // 非SSAA模式下，先渲染模型
        if (m_model->getFaceCount() > 0) {
            m_renderer->renderModel(*m_model);
        }
        
        // 然后绘制其他元素 - 根据开关决定是否绘制
        if (m_renderer->getDrawAxesAndGrid()) {
            m_renderer->drawGrid(5.0f, 5);
            m_renderer->drawAxes(2.0f);
        }
        m_renderer->drawAllLightPositions();
        
        if (m_model->getFaceCount() > 0) {
            if (m_renderer->getDrawLightRays()) {
                m_renderer->drawLightRays(*m_model);
            }
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
    const char* newText = (!currentState) ? "Edge" : "Edge";
    SetWindowTextA(m_toggleEdgesBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleRays() {
    bool currentState = m_renderer->getDrawLightRays();
    m_renderer->setDrawLightRays(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Ray" : "Ray";
    SetWindowTextA(m_toggleRaysBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleTexture() {
    bool currentState = m_renderer->isTextureEnabled();
    m_renderer->setTextureEnabled(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Tex" : "Tex";
    SetWindowTextA(m_toggleTextureBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleNormalMap() {
    bool currentState = m_renderer->isNormalMapEnabled();
    m_renderer->setNormalMapEnabled(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Norm" : "Norm";
    SetWindowTextA(m_toggleNormalMapBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleAxesGrid() {
    bool currentState = m_renderer->getDrawAxesAndGrid();
    m_renderer->setDrawAxesAndGrid(!currentState);
    
    // 更新按钮文本
    const char* newText = (!currentState) ? "Grid" : "Grid";
    SetWindowTextA(m_toggleAxesGridBtn, newText);
    
    UpdateRender();
}

void RenderWindow::OnToggleSSAA() {
    bool currentState = m_renderer->isSSAAEnabled();
    
    if (currentState) {
        m_renderer->disableSSAA();
        SetWindowTextA(m_toggleSSAABtn, "OFF");
    } else {
        m_renderer->enableSSAA(true, m_renderer->getSSAAScale());
        SetWindowTextA(m_toggleSSAABtn, "ON");
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
        sprintf_s(statusText, sizeof(statusText), "ON %dx", scale);
    } else {
        sprintf_s(statusText, sizeof(statusText), "OFF %dx", scale);
    }
    
    SetWindowTextA(m_ssaaStatusLabel, statusText);
    
    // 更新按钮状态
    EnableWindow(m_ssaaScaleIncBtn, scale < 8);
    EnableWindow(m_ssaaScaleDecBtn, scale > 2);
}

void RenderWindow::OnLoadModel() {
    // 获取输入框中的文件名
    char buffer[256];
    GetWindowTextA(m_modelFileEdit, buffer, sizeof(buffer));
    
    if (strlen(buffer) == 0) {
        SetWindowTextA(m_modelStatusLabel, "Error: Empty");
        return;
    }
    
    std::string filename = buffer;
    std::string modelPath = "assets/" + filename;
    
    // 创建新的模型对象
    auto newModel = std::make_unique<Model>();
    
    if (newModel->loadFromFile(modelPath)) {
        // 加载成功，替换当前模型
        newModel->centerModel();
        newModel->scaleModel(0.5f);
        
        m_model = std::move(newModel);
        m_currentModelFile = filename;
        
        // 更新状态标签
        SetWindowTextA(m_modelStatusLabel, filename.c_str());
        
        std::cout << "成功加载模型: " << modelPath << std::endl;
        std::cout << "模型包含 " << m_model->getFaceCount() << " 个面" << std::endl;
        
        // 重新渲染
        UpdateRender();
    } else {
        // 加载失败
        std::string statusText = "Error: " + filename;
        SetWindowTextA(m_modelStatusLabel, statusText.c_str());
        
        std::cout << "加载模型失败: " << modelPath << std::endl;
        std::cout << "请确认文件存在于assets文件夹中" << std::endl;
    }
}

void RenderWindow::OnLoadTexture() {
    // 获取输入框中的文件名
    char buffer[256];
    GetWindowTextA(m_textureFileEdit, buffer, sizeof(buffer));
    
    if (strlen(buffer) == 0) {
        SetWindowTextA(m_textureStatusLabel, "Error: Empty");
        return;
    }
    
    std::string filename = buffer;
    std::string texturePath = "assets/" + filename;
    
    // 创建新的纹理对象
    auto newTexture = std::make_shared<Texture>();
    
    if (newTexture->loadFromFile(texturePath)) {
        // 加载成功，替换当前纹理
        m_texture = newTexture;
        m_renderer->setTexture(m_texture);
        m_currentTextureFile = filename;
        
        // 更新状态标签
        SetWindowTextA(m_textureStatusLabel, filename.c_str());
        
        std::cout << "成功加载纹理: " << texturePath << std::endl;
        
        // 重新渲染
        UpdateRender();
    } else {
        // 加载失败
        std::string statusText = "Error: " + filename;
        SetWindowTextA(m_textureStatusLabel, statusText.c_str());
        
        std::cout << "加载纹理失败: " << texturePath << std::endl;
        std::cout << "请确认文件存在于assets文件夹中且为BMP格式" << std::endl;
    }
}

void RenderWindow::OnLoadNormalMap() {
    // 获取输入框中的文件名
    char buffer[256];
    GetWindowTextA(m_normalMapFileEdit, buffer, sizeof(buffer));
    
    if (strlen(buffer) == 0) {
        SetWindowTextA(m_normalMapStatusLabel, "Error: Empty");
        return;
    }
    
    std::string filename = buffer;
    std::string normalMapPath = "assets/" + filename;
    
    // 创建新的法线贴图对象
    auto newNormalMap = std::make_shared<Texture>();
    
    if (newNormalMap->loadFromFile(normalMapPath)) {
        // 加载成功，替换当前法线贴图
        m_normalMap = newNormalMap;
        m_renderer->setNormalMap(m_normalMap);
        m_currentNormalMapFile = filename;
        
        // 更新状态标签
        SetWindowTextA(m_normalMapStatusLabel, filename.c_str());
        
        std::cout << "成功加载法线贴图: " << normalMapPath << std::endl;
        
        // 重新渲染
        UpdateRender();
    } else {
        // 加载失败
        std::string statusText = "Error: " + filename;
        SetWindowTextA(m_normalMapStatusLabel, statusText.c_str());
        
        std::cout << "加载法线贴图失败: " << normalMapPath << std::endl;
        std::cout << "请确认文件存在于assets文件夹中且为BMP格式" << std::endl;
    }
}

void RenderWindow::OnBRDFParameterChanged() {
    char buffer[32];
    
    GetWindowTextA(m_roughnessEdit, buffer, sizeof(buffer));
    m_roughness = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_metallicEdit, buffer, sizeof(buffer));
    m_metallic = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_energyCompensationScaleEdit, buffer, sizeof(buffer));
    m_energyCompensationScale = static_cast<float>(atof(buffer));
    
    m_renderer->setRoughness(m_roughness);
    m_renderer->setMetallic(m_metallic);
    m_renderer->setEnergyCompensationScale(m_energyCompensationScale);
    
    UpdateRender();
} 