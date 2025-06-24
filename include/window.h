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
    
    // 新增：FOV控制方法
    void OnFovIncrease();
    void OnFovDecrease();
    
    // 新增：绘制控制方法
    void OnToggleEdges();
    void OnToggleRays();
    
    // 新增：SSAA控制方法
    void OnToggleSSAA();
    void OnSSAAScaleIncrease();
    void OnSSAAScaleDecrease();
    
    void OnCameraChanged();
    void OnRotationChanged();
    void OnLightChanged();
    void OnLight2Changed(); // 新增：第二个光源变化处理
    void OnLightingChanged(); // 新增：光照系数变化处理
    void RenderToWindow();
    
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    void CreateControls();
    void UpdateSSAAControls();
    
    // Window handles
    HWND m_hwnd;
    HWND m_renderArea;
    HWND m_cameraXEdit, m_cameraYEdit, m_cameraZEdit;
    HWND m_rotationXEdit, m_rotationYEdit, m_rotationZEdit;
    HWND m_cameraRollXEdit, m_cameraRollYEdit, m_cameraRollZEdit; // 摄像机XYZ轴旋转
    HWND m_lightXEdit, m_lightYEdit, m_lightZEdit, m_lightIntensityEdit;
    HWND m_light2XEdit, m_light2YEdit, m_light2ZEdit, m_light2IntensityEdit; // 第二个光源
    HWND m_cameraLabel, m_rotationLabel, m_cameraRollLabel, m_lightLabel, m_light2Label;
    
    // 新增：光照系数控件
    HWND m_diffuseEdit, m_specularEdit, m_ambientEdit;
    HWND m_lightingLabel;
    
    // 新增：FOV控制按钮
    HWND m_fovIncreaseBtn, m_fovDecreaseBtn;
    
    // 新增：绘制控制按钮
    HWND m_toggleEdgesBtn, m_toggleRaysBtn;
    
    // 新增：SSAA控制按钮和标签
    HWND m_toggleSSAABtn, m_ssaaScaleIncBtn, m_ssaaScaleDecBtn;
    HWND m_ssaaStatusLabel;
    
    // Rendering
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Model> m_model;
    std::shared_ptr<Texture> m_texture;
    
    // Window properties
    int m_windowWidth, m_windowHeight;
    int m_renderWidth, m_renderHeight;
    
    // Camera settings
    float m_cameraX, m_cameraY, m_cameraZ;
    float m_rotationX, m_rotationY, m_rotationZ;
    float m_cameraRollX, m_cameraRollY, m_cameraRollZ; // 摄像机XYZ轴旋转
    
    // 新增：FOV设置
    float m_fov;
    
    // Light settings
    float m_lightX, m_lightY, m_lightZ, m_lightIntensity;
    float m_light2X, m_light2Y, m_light2Z, m_light2Intensity; // 第二个光源设置
    
    // 新增：光照系数设置
    float m_diffuseStrength, m_specularStrength, m_ambientStrength;
    
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
    static const int ID_ROTATION_Z = 1013;
    static const int ID_CAMERA_ROLL_X = 1014;
    static const int ID_CAMERA_ROLL_Y = 1015;
    static const int ID_CAMERA_ROLL_Z = 1016;
    static const int ID_LIGHT_X = 1006;
    static const int ID_LIGHT_Y = 1007;
    static const int ID_LIGHT_Z = 1008;
    static const int ID_LIGHT_INTENSITY = 1009;
    
    // 新增：FOV控制ID
    static const int ID_FOV_INCREASE = 1011;
    static const int ID_FOV_DECREASE = 1012;
    
    // 新增：绘制控制ID
    static const int ID_TOGGLE_EDGES = 1017;
    static const int ID_TOGGLE_RAYS = 1018;
    
    // 新增：SSAA控制ID
    static const int ID_TOGGLE_SSAA = 1019;
    static const int ID_SSAA_SCALE_INC = 1020;
    static const int ID_SSAA_SCALE_DEC = 1021;
    
    // 新增：第二个光源控制ID
    static const int ID_LIGHT2_X = 1022;
    static const int ID_LIGHT2_Y = 1023;
    static const int ID_LIGHT2_Z = 1024;
    static const int ID_LIGHT2_INTENSITY = 1025;
    
    // 新增：光照系数控制ID
    static const int ID_DIFFUSE_STRENGTH = 1026;
    static const int ID_SPECULAR_STRENGTH = 1027;
    static const int ID_AMBIENT_STRENGTH = 1028;
}; 