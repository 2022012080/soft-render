#include "window.h"
#include "vector_math.h"
#include <commctrl.h>
#include <windowsx.h>
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
    , m_dirLightX(0.0f), m_dirLightY(-1.0f), m_dirLightZ(0.0f), m_dirLightIntensity(5.0f) // 平面光源：向下照射
    , m_diffuseStrength(0.2f), m_specularStrength(2.0f), m_ambientStrength(0.15f) // 光照系数
    , m_shininess(128.0f) // 新增：高光指数初始化
    , m_hwnd(nullptr), m_renderArea(nullptr)
    , m_bitmap(nullptr), m_memDC(nullptr), m_bitmapData(nullptr)
    , m_currentModelFile("sphere.obj") // 默认模型文件
    , m_currentTextureFile("texture.bmp") // 默认纹理文件
    , m_currentNormalMapFile("normal.bmp") // 默认法线贴图文件
    , m_roughness(0.5f)        // 新增：粗糙度初始化
    , m_metallic(0.0f)         // 新增：金属度初始化
    , m_f0R(0.04f), m_f0G(0.04f), m_f0B(0.04f)  // 新增：菲涅尔F0初始化
    , m_energyCompensationScale(1.0f)  // 新增：能量补偿强度初始化
    , m_cameraPos(0.0f, 0.0f, 0.0f)
    , m_cameraTargetPos(0.0f, 0.0f, 0.0f)
    , m_cameraMoveSpeed(0.15f)
    , m_fovTarget(20.0f)
    , m_fovLerpSpeed(0.15f)
    , m_dragging(false)
    , m_lastMousePos{0, 0}
    , m_cameraYaw(0.0f)
    , m_cameraPitch(0.0f)
    , m_moveSpeed(0.15f)  // 新增：移动速度初始化
    , m_cameraRotation()  // 初始化为单位四元数
    , m_targetRotation()
    , m_rotationLerpSpeed(0.15f)
    , m_targetPosition(0.0f, 0.0f, -5.0f)
    , m_moveLerpSpeed(0.15f)
    , m_isMoving(false)
{
    // 初始化键盘状态数组
    memset(m_keyStates, 0, sizeof(m_keyStates));
    
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
    OnDirLightChanged();  // 新增：平面光源
    OnLightingChanged();  // 光照系数
    OnBRDFParameterChanged();  // 新增：BRDF参数包括F0
    
    // 设置能量补偿复选框的默认状态为选中
    SendMessage(m_energyCompensationCheckbox, BM_SETCHECK, BST_CHECKED, 0);
    
    // Initial render
    UpdateRender();
    
    ShowWindow(m_hwnd, SW_SHOW);
    UpdateWindow(m_hwnd);
    
    StartMoveTimer();
    
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
    
    // Directional Light controls
    m_dirLightLabel = CreateWindowA("STATIC", "DirLight (X,Y,Z):", WS_VISIBLE | WS_CHILD,
        1220, 210, 100, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_dirLightXEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 225, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIRLIGHT_X, GetModuleHandle(nullptr), nullptr);
    
    m_dirLightYEdit = CreateWindowA("EDIT", "-1.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 225, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIRLIGHT_Y, GetModuleHandle(nullptr), nullptr);
    
    m_dirLightZEdit = CreateWindowA("EDIT", "0.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 225, 45, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIRLIGHT_Z, GetModuleHandle(nullptr), nullptr);
    
    // Directional Light intensity control
    CreateWindowA("STATIC", "Intensity:", WS_VISIBLE | WS_CHILD,
        1370, 210, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_dirLightIntensityEdit = CreateWindowA("EDIT", "5.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1370, 225, 50, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIRLIGHT_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
    // Lighting coefficients
    m_lightingLabel = CreateWindowA("STATIC", "Lighting:", WS_VISIBLE | WS_CHILD,
        1220, 250, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Diff:", WS_VISIBLE | WS_CHILD,
        1220, 265, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_diffuseEdit = CreateWindowA("EDIT", "0.2", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 280, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_DIFFUSE_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Spec:", WS_VISIBLE | WS_CHILD,
        1270, 265, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_specularEdit = CreateWindowA("EDIT", "2.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 280, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_SPECULAR_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    CreateWindowA("STATIC", "Amb:", WS_VISIBLE | WS_CHILD,
        1320, 265, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_ambientEdit = CreateWindowA("EDIT", "0.15", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 280, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_AMBIENT_STRENGTH, GetModuleHandle(nullptr), nullptr);
    
    // Shininess control
    CreateWindowA("STATIC", "Shin:", WS_VISIBLE | WS_CHILD,
        1370, 265, 35, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    m_shininessEdit = CreateWindowA("EDIT", "128.0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1370, 280, 50, 20, m_hwnd, (HMENU)(LONG_PTR)ID_SHININESS, GetModuleHandle(nullptr), nullptr);
    
    // FOV controls
    CreateWindowA("STATIC", "FOV:", WS_VISIBLE | WS_CHILD,
        1220, 300, 30, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_fovDecreaseBtn = CreateWindowA("BUTTON", "(-)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 315, 30, 25, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_DECREASE, GetModuleHandle(nullptr), nullptr);
    
    m_fovIncreaseBtn = CreateWindowA("BUTTON", "(+)", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1255, 315, 30, 25, m_hwnd, (HMENU)(LONG_PTR)ID_FOV_INCREASE, GetModuleHandle(nullptr), nullptr);
    
    // Render controls
    CreateWindowA("STATIC", "Render:", WS_VISIBLE | WS_CHILD,
        1220, 350, 50, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleEdgesBtn = CreateWindowA("BUTTON", "Edge", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 365, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_EDGES, GetModuleHandle(nullptr), nullptr);
    
    m_toggleRaysBtn = CreateWindowA("BUTTON", "Ray", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1265, 365, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_RAYS, GetModuleHandle(nullptr), nullptr);
    
    // Texture controls
    m_toggleTextureBtn = CreateWindowA("BUTTON", "Tex", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1310, 365, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_TEXTURE, GetModuleHandle(nullptr), nullptr);
    
    // Normal map control
    m_toggleNormalMapBtn = CreateWindowA("BUTTON", "Norm", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1355, 365, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_NORMAL_MAP, GetModuleHandle(nullptr), nullptr);
    
    // Axes/Grid control
    m_toggleAxesGridBtn = CreateWindowA("BUTTON", "Grid", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1400, 365, 40, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_AXES_GRID, GetModuleHandle(nullptr), nullptr);
    
    // SSAA controls
    CreateWindowA("STATIC", "SSAA:", WS_VISIBLE | WS_CHILD,
        1220, 395, 40, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_toggleSSAABtn = CreateWindowA("BUTTON", "OFF", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1220, 410, 35, 25, m_hwnd, (HMENU)(LONG_PTR)ID_TOGGLE_SSAA, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleDecBtn = CreateWindowA("BUTTON", "-", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1260, 410, 25, 25, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_DEC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaScaleIncBtn = CreateWindowA("BUTTON", "+", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1290, 410, 25, 25, m_hwnd, (HMENU)(LONG_PTR)ID_SSAA_SCALE_INC, GetModuleHandle(nullptr), nullptr);
    
    m_ssaaStatusLabel = CreateWindowA("STATIC", "OFF", WS_VISIBLE | WS_CHILD,
        1320, 395, 100, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Model file controls
    CreateWindowA("STATIC", "Model:", WS_VISIBLE | WS_CHILD,
        1220, 440, 50, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_modelFileEdit = CreateWindowA("EDIT", "sphere.obj", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 455, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_MODEL_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadModelBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 455, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_MODEL, GetModuleHandle(nullptr), nullptr);
    
    m_modelStatusLabel = CreateWindowA("STATIC", "sphere.obj", WS_VISIBLE | WS_CHILD,
        1390, 455, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Texture file controls
    CreateWindowA("STATIC", "Texture:", WS_VISIBLE | WS_CHILD,
        1220, 480, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_textureFileEdit = CreateWindowA("EDIT", "texture.bmp", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 495, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_TEXTURE_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadTextureBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 495, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_TEXTURE, GetModuleHandle(nullptr), nullptr);
    
    m_textureStatusLabel = CreateWindowA("STATIC", "texture.bmp", WS_VISIBLE | WS_CHILD,
        1390, 495, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Normal map file controls
    CreateWindowA("STATIC", "Normal:", WS_VISIBLE | WS_CHILD,
        1220, 520, 60, 15, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_normalMapFileEdit = CreateWindowA("EDIT", "normal.bmp", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 535, 120, 20, m_hwnd, (HMENU)(LONG_PTR)ID_NORMAL_MAP_FILE, GetModuleHandle(nullptr), nullptr);
    
    m_loadNormalMapBtn = CreateWindowA("BUTTON", "Load", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        1345, 535, 40, 20, m_hwnd, (HMENU)(LONG_PTR)ID_LOAD_NORMAL_MAP, GetModuleHandle(nullptr), nullptr);
    
    m_normalMapStatusLabel = CreateWindowA("STATIC", "normal.bmp", WS_VISIBLE | WS_CHILD,
        1390, 535, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // Render area
    m_renderArea = CreateWindowA("STATIC", "", WS_VISIBLE | WS_CHILD | WS_BORDER | SS_BITMAP,
        10, 10, m_renderWidth, m_renderHeight, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    // BRDF Model controls
    CreateWindowA("STATIC", "BRDF Model:", WS_VISIBLE | WS_CHILD,
        1220, 560, 80, 15, m_hwnd, NULL, NULL, NULL);
    
    m_brdfCheckbox = CreateWindowA("BUTTON", "Enable BRDF", 
                                   WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                   1220, 575, 100, 20, m_hwnd, 
                                   (HMENU)ID_BRDF_ENABLE, NULL, NULL);
    
    // Roughness control
    CreateWindowA("STATIC", "Rough:", WS_VISIBLE | WS_CHILD,
        1220, 600, 45, 15, m_hwnd, NULL, NULL, NULL);
    
    m_roughnessEdit = CreateWindowA("EDIT", "0.5", 
                                    WS_VISIBLE | WS_CHILD | WS_BORDER,
                                    1270, 600, 50, 20, m_hwnd, 
                                    (HMENU)ID_ROUGHNESS, NULL, NULL);
    
    // Metallic control
    CreateWindowA("STATIC", "Metal:", WS_VISIBLE | WS_CHILD,
        1330, 600, 40, 15, m_hwnd, NULL, NULL, NULL);
    
    m_metallicEdit = CreateWindowA("EDIT", "0.0", 
                                   WS_VISIBLE | WS_CHILD | WS_BORDER,
                                   1375, 600, 50, 20, m_hwnd, 
                                   (HMENU)ID_METALLIC, NULL, NULL);
    
    // 能量补偿控件
    m_energyCompensationCheckbox = CreateWindowA("BUTTON", "Energy Compensation", 
                                                  WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                                  1220, 625, 140, 20, m_hwnd, 
                                                  (HMENU)ID_ENERGY_COMPENSATION_ENABLE, NULL, NULL);
    
    CreateWindowA("STATIC", "EC Scale:", WS_VISIBLE | WS_CHILD,
        1220, 650, 60, 15, m_hwnd, NULL, NULL, NULL);
    
    m_energyCompensationScaleEdit = CreateWindowA("EDIT", "1.0", 
                                                   WS_VISIBLE | WS_CHILD | WS_BORDER,
                                                   1285, 650, 50, 20, m_hwnd, 
                                                   (HMENU)ID_ENERGY_COMPENSATION_SCALE, NULL, NULL);
    
    // 新增：菲涅尔F0控件
    CreateWindowA("STATIC", "Fresnel F0 (R,G,B):", WS_VISIBLE | WS_CHILD,
        1220, 675, 120, 15, m_hwnd, NULL, NULL, NULL);
    
    m_f0REdit = CreateWindowA("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1220, 690, 45, 20, m_hwnd, (HMENU)ID_F0_R, NULL, NULL);
    
    m_f0GEdit = CreateWindowA("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1270, 690, 45, 20, m_hwnd, (HMENU)ID_F0_G, NULL, NULL);
    
    m_f0BEdit = CreateWindowA("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER,
        1320, 690, 45, 20, m_hwnd, (HMENU)ID_F0_B, NULL, NULL);
    
    // Displacement shader controls
    int displacementY = 715;
    m_displacementLabel = CreateWindowA("STATIC", "Displacement Shader", 
                                     WS_CHILD | WS_VISIBLE,
                                     1220, displacementY, 150, 15,
                                     m_hwnd, NULL, NULL, NULL);
    displacementY += 20;
    
    m_toggleDisplacementBtn = CreateWindowA("BUTTON", "Enable Displacement",
                                         WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                         1220, displacementY, 150, 20,
                                         m_hwnd, (HMENU)ID_TOGGLE_DISPLACEMENT, NULL, NULL);
    displacementY += 25;
    
    CreateWindowA("STATIC", "Scale:", WS_CHILD | WS_VISIBLE,
                1220, displacementY, 45, 15, m_hwnd, NULL, NULL, NULL);
    m_displacementScaleEdit = CreateWindowA("EDIT", "0.5",
                                         WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                         1270, displacementY, 45, 20,
                                         m_hwnd, (HMENU)ID_DISPLACEMENT_SCALE, NULL, NULL);
    displacementY += 25;
    
    CreateWindowA("STATIC", "Frequency:", WS_CHILD | WS_VISIBLE,
                1220, displacementY, 45, 15, m_hwnd, NULL, NULL, NULL);
    m_displacementFreqEdit = CreateWindowA("EDIT", "8.0",
                                        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                        1270, displacementY, 45, 20,
                                        m_hwnd, (HMENU)ID_DISPLACEMENT_FREQ, NULL, NULL);
    displacementY += 25;
    
    CreateWindowA("STATIC", "Length:", WS_CHILD | WS_VISIBLE,
                1220, displacementY, 45, 15, m_hwnd, NULL, NULL, NULL);
    m_spineLengthEdit = CreateWindowA("EDIT", "0.2",
                                   WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                   1270, displacementY, 45, 20,
                                   m_hwnd, (HMENU)ID_SPINE_LENGTH, NULL, NULL);
    displacementY += 25;
    
    CreateWindowA("STATIC", "Sharpness:", WS_CHILD | WS_VISIBLE,
                1220, displacementY, 45, 15, m_hwnd, NULL, NULL, NULL);
    m_spineSharpnessEdit = CreateWindowA("EDIT", "2.0",
                                      WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                      1270, displacementY, 45, 20,
                                      m_hwnd, (HMENU)ID_SPINE_SHARPNESS, NULL, NULL);
    displacementY += 25;

    // 摄像机角度显示控件（放在右侧靠下，不与其他控件重叠）
    m_cameraAngleLabel = CreateWindowA("STATIC", "Yaw: 0.0  Pitch: 0.0", WS_VISIBLE | WS_CHILD,
        1220, 750, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    // 新增：摄像机旋转显示控件（放在摄像机角度显示控件上方）
    m_cameraRotationLabel = CreateWindowA("STATIC", "Roll: 0.0  Pitch: 0.0  Yaw: 0.0", WS_VISIBLE | WS_CHILD,
        1220, 725, 200, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
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
        StopMoveTimer();
        PostQuitMessage(0);
        return 0;
        
    case WM_COMMAND:
        if (HIWORD(wParam) == EN_CHANGE) {
            int controlId = LOWORD(wParam);
            if (controlId >= ID_ROTATION_X && controlId <= ID_ROTATION_Y || controlId == ID_ROTATION_Z) {
                OnRotationChanged();
            } else if (controlId >= ID_CAMERA_ROLL_X && controlId <= ID_CAMERA_ROLL_Z) {
                OnCameraChanged();  // 摄像机roll角度变化也调用OnCameraChanged
            } else if (controlId >= ID_LIGHT_X && controlId <= ID_LIGHT_INTENSITY) {
                OnLightChanged();
            } else if (controlId >= ID_LIGHT2_X && controlId <= ID_LIGHT2_INTENSITY) {
                OnLight2Changed();
            } else if (controlId >= ID_DIRLIGHT_X && controlId <= ID_DIRLIGHT_INTENSITY) {
                OnDirLightChanged();
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
            } else if (controlId >= ID_F0_R && controlId <= ID_F0_B) {
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
            } else if (controlId == ID_TOGGLE_DISPLACEMENT || controlId == ID_DISPLACEMENT_SCALE ||
                       controlId == ID_DISPLACEMENT_FREQ || controlId == ID_SPINE_LENGTH ||
                       controlId == ID_SPINE_SHARPNESS) {
                OnDisplacementChanged();
            }
        }
        return 0;

    case WM_CHAR:
        if (wParam == VK_RETURN) {
            HWND focusedWindow = GetFocus();
            if (focusedWindow) {
                int controlId = GetDlgCtrlID(focusedWindow);
                if (controlId >= ID_OBJECT_X && controlId <= ID_OBJECT_Z) {
                    OnObjectChanged();
                }
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
    
    case WM_KEYDOWN:
        {
            m_keyStates[wParam & 0xFF] = true;
        }
        break;

    case WM_KEYUP:
        {
            m_keyStates[wParam & 0xFF] = false;
        }
        break;

    case WM_TIMER:
        if (wParam == MOVE_TIMER_ID) {
            UpdateMovement();
        }
        break;
    
    case WM_MOUSEWHEEL:
        {
            short delta = GET_WHEEL_DELTA_WPARAM(wParam);
            m_fov -= delta * 0.01f * 10.0f; // 每格10度，直接更新FOV
            if (m_fov < 10.0f) m_fov = 10.0f;
            if (m_fov > 120.0f) m_fov = 120.0f;
            UpdateRender();
        }
        break;
    
    case WM_LBUTTONDOWN: {
        POINT pt;
        pt.x = GET_X_LPARAM(lParam);
        pt.y = GET_Y_LPARAM(lParam);
        RECT rc;
        GetWindowRect(m_renderArea, &rc);
        ScreenToClient(m_hwnd, (LPPOINT)&rc.left);
        ScreenToClient(m_hwnd, (LPPOINT)&rc.right);
        if (pt.x >= rc.left && pt.x < rc.right && pt.y >= rc.top && pt.y < rc.bottom) {
            m_dragging = true;
            m_lastMousePos = pt;
            SetCapture(m_hwnd);
        }
        SetFocus(m_hwnd);
        break;
    }
    case WM_MOUSEMOVE: {
        if (m_dragging && (wParam & MK_LBUTTON)) {
            POINT pt;
            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);
            int dx = pt.x - m_lastMousePos.x;
            int dy = pt.y - m_lastMousePos.y;
            m_lastMousePos = pt;

            // 修正旋转映射：取反dx和dy
            m_cameraYaw -= dx / 50.0f;
            m_cameraPitch += dy / 50.0f;

            // 限制俯仰角范围
            if (m_cameraPitch > 89.0f) m_cameraPitch = 89.0f;
            if (m_cameraPitch < -89.0f) m_cameraPitch = -89.0f;

            // 将欧拉角转换为四元数时，需要先绕X轴（pitch），再绕Y轴（yaw）
            float yawRad = m_cameraYaw * 3.1415926f / 180.0f;
            float pitchRad = m_cameraPitch * 3.1415926f / 180.0f;
            
            // 创建目标四元数：先pitch后yaw
            VectorMath::Quaternion pitchRotation = VectorMath::Quaternion::fromAxisAngle(Vec3f(1, 0, 0), pitchRad);
            VectorMath::Quaternion yawRotation = VectorMath::Quaternion::fromAxisAngle(Vec3f(0, 1, 0), yawRad);
            m_targetRotation = yawRotation * pitchRotation;
            
            // 在拖动时使用插值
            m_cameraRotation = VectorMath::Quaternion::slerp(m_cameraRotation, m_targetRotation, m_rotationLerpSpeed);
            
            UpdateCameraAngleLabel();
            UpdateRender();
        }
        break;
    }
    case WM_LBUTTONUP:
        if (m_dragging) {
            m_dragging = false;
            // 在松开鼠标时立即应用目标旋转
            m_cameraRotation = m_targetRotation;
            ReleaseCapture();
            UpdateRender();
        }
        break;
    
    default:
        break;
    }
    
    return DefWindowProc(m_hwnd, uMsg, wParam, lParam);
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
    UpdateCameraRotationLabel();
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
    // --- 平滑插值摄像机位置 ---
    m_cameraPos = m_cameraPos + (m_cameraTargetPos - m_cameraPos) * m_cameraMoveSpeed;
    
    // --- 视图矩阵 ---
    Matrix4x4 modelMatrix = VectorMath::translate(Vec3f(m_objectX, m_objectY, m_objectZ)) * 
                           VectorMath::rotate(m_rotationZ, Vec3f(0, 0, 1)) *
                           VectorMath::rotate(m_rotationY, Vec3f(0, 1, 0)) *
                           VectorMath::rotate(m_rotationX, Vec3f(1, 0, 0)) *
                           VectorMath::scale(Vec3f(0.5f, 0.5f, 0.5f));
    // 摄像机位置和朝向
    Vec3f cameraPos = m_cameraPos;
    // 使用球面插值平滑过渡到目标旋转
    m_cameraRotation = VectorMath::Quaternion::slerp(m_cameraRotation, m_targetRotation, m_rotationLerpSpeed);
    
    // 获取旋转矩阵
    Matrix4x4 rotationMatrix = m_cameraRotation.toMatrix();
    
    // 计算摄像机方向
    Vec3f forward = rotationMatrix * Vec3f(0, 0, -1);
    Vec3f target = m_cameraPos + forward;
    Vec3f worldUp(0, 1, 0);
    
    Matrix4x4 viewMatrix = VectorMath::lookAt(cameraPos, target, worldUp);
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
    if (m_renderer->getLightCount() >= 3) {
        // 更新第一个光源（点光源）
        Light& light1 = m_renderer->getLight(0);
        light1.type = LightType::POINT;
        light1.position = Vec3f(m_lightX, m_lightY, m_lightZ);
        light1.color = Vec3f(1, 1, 1); // 白色
        light1.intensity = m_lightIntensity;
        
        // 更新第二个光源（点光源）
        Light& light2 = m_renderer->getLight(1);
        light2.type = LightType::POINT;
        light2.position = Vec3f(m_light2X, m_light2Y, m_light2Z);
        light2.color = Vec3f(0.8, 0.6, 1.0); // 紫色
        light2.intensity = m_light2Intensity;
        
        // 更新第三个光源（平面光源）
        Light& light3 = m_renderer->getLight(2);
        light3.type = LightType::DIRECTIONAL;
        light3.position = Vec3f(m_dirLightX, m_dirLightY, m_dirLightZ).normalize(); // 作为方向向量
        light3.color = Vec3f(1.0, 0.9, 0.8); // 暖白色
        light3.intensity = m_dirLightIntensity;
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
    
    // 使用用户输入的F0值（如果输入框为空则使用默认值）
    m_renderer->setF0(Vec3f(m_f0R, m_f0G, m_f0B));
    
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
    
    // 读取菲涅尔F0值，如果输入框为空则使用默认值
    GetWindowTextA(m_f0REdit, buffer, sizeof(buffer));
    if (strlen(buffer) > 0) {
        m_f0R = static_cast<float>(atof(buffer));
    } else {
        m_f0R = 0.04f;  // 默认值
    }
    
    GetWindowTextA(m_f0GEdit, buffer, sizeof(buffer));
    if (strlen(buffer) > 0) {
        m_f0G = static_cast<float>(atof(buffer));
    } else {
        m_f0G = 0.04f;  // 默认值
    }
    
    GetWindowTextA(m_f0BEdit, buffer, sizeof(buffer));
    if (strlen(buffer) > 0) {
        m_f0B = static_cast<float>(atof(buffer));
    } else {
        m_f0B = 0.04f;  // 默认值
    }
    
    m_renderer->setRoughness(m_roughness);
    m_renderer->setMetallic(m_metallic);
    m_renderer->setEnergyCompensationScale(m_energyCompensationScale);
    m_renderer->setF0(Vec3f(m_f0R, m_f0G, m_f0B));
    
    UpdateRender();
}

void RenderWindow::OnDirLightChanged() {
    // Get directional light values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_dirLightXEdit, buffer, sizeof(buffer));
    m_dirLightX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_dirLightYEdit, buffer, sizeof(buffer));
    m_dirLightY = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_dirLightZEdit, buffer, sizeof(buffer));
    m_dirLightZ = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_dirLightIntensityEdit, buffer, sizeof(buffer));
    m_dirLightIntensity = static_cast<float>(atof(buffer));
    
    UpdateRender();
}

void RenderWindow::OnDisplacementChanged() {
    if (!m_renderer) return;
    
    // 获取启用状态
    bool enabled = SendMessage(m_toggleDisplacementBtn, BM_GETCHECK, 0, 0) == BST_CHECKED;
    m_renderer->setDisplacementEnabled(enabled);
    
    // 获取位移强度
    char buffer[32];
    GetWindowTextA(m_displacementScaleEdit, buffer, sizeof(buffer));
    float scale = std::atof(buffer);
    m_renderer->setDisplacementScale(scale);
    
    // 获取位移频率
    GetWindowTextA(m_displacementFreqEdit, buffer, sizeof(buffer));
    float freq = std::atof(buffer);
    m_renderer->setDisplacementFrequency(freq);
    
    // 获取刺长度
    GetWindowTextA(m_spineLengthEdit, buffer, sizeof(buffer));
    float length = std::atof(buffer);
    m_renderer->setSpineLength(length);
    
    // 获取刺锐利度
    GetWindowTextA(m_spineSharpnessEdit, buffer, sizeof(buffer));
    float sharpness = std::atof(buffer);
    m_renderer->setSpineSharpness(sharpness);
    
    // 更新渲染
    UpdateRender();
}

void RenderWindow::StartMoveTimer() {
    SetTimer(m_hwnd, MOVE_TIMER_ID, MOVE_TIMER_INTERVAL, nullptr);
}

void RenderWindow::StopMoveTimer() {
    KillTimer(m_hwnd, MOVE_TIMER_ID);
}

void RenderWindow::UpdateMovement() {
    bool needsUpdate = false;
    float moveStep = m_moveSpeed * MOVE_TIMER_INTERVAL / 1000.0f * 40.0f;

    // 获取当前摄像机的朝向
    Matrix4x4 rotationMatrix = m_cameraRotation.toMatrix();
    Vec3f forward = rotationMatrix * Vec3f(0, 0, -1);
    Vec3f right = rotationMatrix * Vec3f(1, 0, 0);
    
    forward = forward.normalize();
    right = right.normalize();

    // 检查是否有任何移动键被按下
    bool anyMovementKey = m_keyStates['W'] || m_keyStates['S'] || 
                         m_keyStates['A'] || m_keyStates['D'] || 
                         (m_keyStates[VK_SPACE] && (m_keyStates['W'] || m_keyStates['S']));

    if (!anyMovementKey) {
        // 如果没有按键按下，停止移动插值
        m_isMoving = false;
        m_targetPosition = Vec3f(m_objectX, m_objectY, m_objectZ);
        needsUpdate = true;
    } else {
        // 计算目标位置
        Vec3f currentTarget(m_targetPosition.x, m_targetPosition.y, m_targetPosition.z);
        m_isMoving = true;

        if (m_keyStates['W']) {
            if (m_keyStates[VK_SPACE]) {
                currentTarget.y += moveStep;
            } else {
                currentTarget.x -= forward.x * moveStep;
                currentTarget.y -= forward.y * moveStep;
                currentTarget.z -= forward.z * moveStep;
            }
        }
        if (m_keyStates['S']) {
            if (m_keyStates[VK_SPACE]) {
                currentTarget.y -= moveStep;
            } else {
                currentTarget.x += forward.x * moveStep;
                currentTarget.y += forward.y * moveStep;
                currentTarget.z += forward.z * moveStep;
            }
        }
        if (m_keyStates['A']) {
            currentTarget.x += right.x * moveStep;
            currentTarget.y += right.y * moveStep;
            currentTarget.z += right.z * moveStep;
        }
        if (m_keyStates['D']) {
            currentTarget.x -= right.x * moveStep;
            currentTarget.y -= right.y * moveStep;
            currentTarget.z -= right.z * moveStep;
        }

        m_targetPosition = currentTarget;
        needsUpdate = true;
    }

    if (needsUpdate) {
        // 应用插值移动
        if (m_isMoving) {
            Vec3f currentPos(m_objectX, m_objectY, m_objectZ);
            Vec3f newPos = VectorMath::lerp(currentPos, m_targetPosition, m_moveLerpSpeed);
            m_objectX = newPos.x;
            m_objectY = newPos.y;
            m_objectZ = newPos.z;
        }

        // 更新编辑框显示的值
        SetWindowTextA(m_objectXEdit, std::to_string(m_objectX).c_str());
        SetWindowTextA(m_objectYEdit, std::to_string(m_objectY).c_str());
        SetWindowTextA(m_objectZEdit, std::to_string(m_objectZ).c_str());
        UpdateRender();
    }
}

void RenderWindow::UpdateCameraAngleLabel() {
    char buf[128];
    sprintf_s(buf, sizeof(buf), "Yaw: %.1f  Pitch: %.1f", m_cameraYaw, m_cameraPitch);
    SetWindowTextA(m_cameraAngleLabel, buf);
}

void RenderWindow::UpdateCameraRotationLabel() {
    char buf[128];
    sprintf_s(buf, sizeof(buf), "Roll: %.1f  Pitch: %.1f  Yaw: %.1f", m_cameraRollX, m_cameraRollY, m_cameraRollZ);
    SetWindowTextA(m_cameraRotationLabel, buf);
} 