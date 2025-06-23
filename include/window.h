#pragma once

#include <windows.h>
#include <string>
#include <memory>
#include "renderer.h"
#include "model.h"
#include "texture.h"

class RenderWindow {
public:
    RenderWindow(int width, int height);
    ~RenderWindow();
    
    bool Initialize();
    void Run();
    void UpdateRender();
    
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    void CreateControls();
    void OnCameraChanged();
    void OnRotationChanged();
    void OnLightChanged();
    void RenderToWindow();
    
    // Window handles
    HWND m_hwnd;
    HWND m_renderArea;
    HWND m_cameraXEdit, m_cameraYEdit, m_cameraZEdit;
    HWND m_rotationXEdit, m_rotationYEdit;
    HWND m_lightXEdit, m_lightYEdit, m_lightZEdit, m_lightIntensityEdit;
    HWND m_cameraLabel, m_rotationLabel, m_lightLabel;
    
    // Rendering
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Model> m_model;
    std::shared_ptr<Texture> m_texture;
    
    // Window properties
    int m_windowWidth, m_windowHeight;
    int m_renderWidth, m_renderHeight;
    
    // Camera settings
    float m_cameraX, m_cameraY, m_cameraZ;
    float m_rotationX, m_rotationY;
    
    // Light settings
    float m_lightX, m_lightY, m_lightZ, m_lightIntensity;
    
    // DIB for displaying
    HBITMAP m_bitmap;
    HDC m_memDC;
    void* m_bitmapData;
    
    // Control IDs
    static const int ID_CAMERA_X = 1001;
    static const int ID_CAMERA_Y = 1002;
    static const int ID_CAMERA_Z = 1003;
    static const int ID_ROTATION_X = 1004;
    static const int ID_ROTATION_Y = 1005;
    static const int ID_LIGHT_X = 1006;
    static const int ID_LIGHT_Y = 1007;
    static const int ID_LIGHT_Z = 1008;
    static const int ID_LIGHT_INTENSITY = 1009;
}; 