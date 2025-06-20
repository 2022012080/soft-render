#include "window.h"
#include "vector_math.h"
#include <commctrl.h>
#include <iostream>
#include <sstream>

RenderWindow::RenderWindow(int width, int height) 
    : m_windowWidth(width), m_windowHeight(height)
    , m_renderWidth(600), m_renderHeight(450)
    , m_cameraX(0.0f), m_cameraY(0.0f), m_cameraZ(5.0f)
    , m_rotationX(0.0f), m_rotationY(30.0f)
    , m_lightX(3.0f), m_lightY(3.0f), m_lightZ(3.0f), m_lightIntensity(10.0f)
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
    
    m_texture->createDefault(256, 256);
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
    // Camera controls
    CreateWindowA("STATIC", "Camera Position (X, Y, Z):", WS_VISIBLE | WS_CHILD,
        620, 20, 160, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_cameraXEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        620, 45, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_X, GetModuleHandle(nullptr), nullptr);
    
    m_cameraYEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        680, 45, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_Y, GetModuleHandle(nullptr), nullptr);
    
    m_cameraZEdit = CreateWindowA("EDIT", "5", WS_VISIBLE | WS_CHILD | WS_BORDER,
        740, 45, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_CAMERA_Z, GetModuleHandle(nullptr), nullptr);
    
    // Rotation controls
    CreateWindowA("STATIC", "Rotation (Horizontal, Vertical):", WS_VISIBLE | WS_CHILD,
        620, 90, 160, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_rotationXEdit = CreateWindowA("EDIT", "0", WS_VISIBLE | WS_CHILD | WS_BORDER,
        620, 115, 70, 25, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_X, GetModuleHandle(nullptr), nullptr);
    
    m_rotationYEdit = CreateWindowA("EDIT", "30", WS_VISIBLE | WS_CHILD | WS_BORDER,
        700, 115, 70, 25, m_hwnd, (HMENU)(LONG_PTR)ID_ROTATION_Y, GetModuleHandle(nullptr), nullptr);
    
    // Light controls
    CreateWindowA("STATIC", "Light Position (X, Y, Z):", WS_VISIBLE | WS_CHILD,
        620, 160, 160, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightXEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        620, 185, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_X, GetModuleHandle(nullptr), nullptr);
    
    m_lightYEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        680, 185, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Y, GetModuleHandle(nullptr), nullptr);
    
    m_lightZEdit = CreateWindowA("EDIT", "3", WS_VISIBLE | WS_CHILD | WS_BORDER,
        740, 185, 50, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_Z, GetModuleHandle(nullptr), nullptr);
    
    // Light intensity control
    CreateWindowA("STATIC", "Light Intensity:", WS_VISIBLE | WS_CHILD,
        620, 230, 100, 20, m_hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
    
    m_lightIntensityEdit = CreateWindowA("EDIT", "10", WS_VISIBLE | WS_CHILD | WS_BORDER,
        620, 255, 70, 25, m_hwnd, (HMENU)(LONG_PTR)ID_LIGHT_INTENSITY, GetModuleHandle(nullptr), nullptr);
    
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
            } else if (controlId >= ID_ROTATION_X && controlId <= ID_ROTATION_Y) {
                OnRotationChanged();
            } else if (controlId >= ID_LIGHT_X && controlId <= ID_LIGHT_INTENSITY) {
                OnLightChanged();
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
    
    UpdateRender();
}

void RenderWindow::OnRotationChanged() {
    // Get rotation values from edit controls
    char buffer[32];
    
    GetWindowTextA(m_rotationXEdit, buffer, sizeof(buffer));
    m_rotationX = static_cast<float>(atof(buffer));
    
    GetWindowTextA(m_rotationYEdit, buffer, sizeof(buffer));
    m_rotationY = static_cast<float>(atof(buffer));
    
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

void RenderWindow::UpdateRender() {
    // Set up matrices
    Matrix4x4 modelMatrix = VectorMath::translate(Vec3f(0, 0, -5)) * 
                           VectorMath::rotate(m_rotationY, Vec3f(0, 1, 0)) * 
                           VectorMath::rotate(m_rotationX, Vec3f(1, 0, 0)) * 
                           VectorMath::scale(Vec3f(0.5f, 0.5f, 0.5f));
    
    Matrix4x4 viewMatrix = VectorMath::lookAt(
        Vec3f(m_cameraX, m_cameraY, m_cameraZ),
        Vec3f(0, 0, 0),
        Vec3f(0, 1, 0)
    );
    
    Matrix4x4 projectionMatrix = VectorMath::perspective(20.0f, 
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
    
    m_renderer->setLightPosition(Vec3f(m_lightX, m_lightY, m_lightZ));
    m_renderer->setLightColor(Vec3f(1, 1, 1));
    m_renderer->setLightIntensity(m_lightIntensity);
    m_renderer->setAmbientIntensity(0.3f);
    
    // Clear and render
    m_renderer->clear(Color(50, 50, 100));
    m_renderer->clearDepth();
    
    if (m_model->getFaceCount() > 0) {
        m_renderer->renderModel(*m_model);
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