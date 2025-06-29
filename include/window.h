#pragma once

#include <windows.h>
#include <string>
#include <memory>
#include "renderer.h"
#include "model.h"
#include "texture.h"
#include "vector_math.h"

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
    
    // 新增：纹理控制方法
    void OnToggleTexture();
    
    // 新增：法线贴图控制方法
    void OnToggleNormalMap();
    
    // 新增：坐标轴和网格线控制方法
    void OnToggleAxesGrid();
    
    // 新增：SSAA控制方法
    void OnToggleSSAA();
    void OnSSAAScaleIncrease();
    void OnSSAAScaleDecrease();
    
    // 新增：模型加载方法
    void OnLoadModel();
    
    // 新增：纹理加载方法
    void OnLoadTexture();
    
    // 新增：法线贴图加载方法
    void OnLoadNormalMap();
    
    void OnCameraChanged();
    void OnObjectChanged(); // 新增：物体坐标变化处理
    void OnRotationChanged();
    void OnLightChanged();
    void OnLight2Changed(); // 新增：第二个光源变化处理
    void OnLightingChanged(); // 新增：光照系数变化处理
    void OnBRDFParameterChanged(); // 新增：BRDF参数变化处理
    void OnDirLightChanged(); // 新增：平面光源变化处理
    void RenderToWindow();
    
    // 新增：位移着色器控件
    HWND m_toggleDisplacementBtn;
    HWND m_displacementScaleEdit, m_displacementFreqEdit;
    HWND m_spineLengthEdit, m_spineSharpnessEdit;
    HWND m_displacementLabel;
    
    // 新增：位移着色器控制方法
    void OnDisplacementChanged();
    
    HWND m_cameraAngleLabel; // 新增：摄像机角度显示控件
    HWND m_cameraRotationLabel; // 新增：摄像机旋转显示控件
    void UpdateCameraAngleLabel();
    void UpdateCameraRotationLabel(); // 新增：摄像机旋转显示方法
    
    // 新增：自发光控件
    HWND m_toggleEmissionBtn;
    HWND m_emissionStrengthEdit;
    HWND m_emissionColorREdit, m_emissionColorGEdit, m_emissionColorBEdit;
    HWND m_emissionLabel;
    
    // 新增：自发光控制方法
    void OnEmissionChanged();
    
    // Phong参数输入框（RGB）
    HWND m_ambientREdit, m_ambientGEdit, m_ambientBEdit;
    HWND m_diffuseREdit, m_diffuseGEdit, m_diffuseBEdit;
    HWND m_specularREdit, m_specularGEdit, m_specularBEdit;
    
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    void CreateControls();
    void UpdateSSAAControls();
    
    // Window handles
    HWND m_hwnd;
    HWND m_renderArea;
    HWND m_objectXEdit, m_objectYEdit, m_objectZEdit;
    HWND m_rotationXEdit, m_rotationYEdit, m_rotationZEdit;
    HWND m_cameraRollXEdit, m_cameraRollYEdit, m_cameraRollZEdit; // 摄像机XYZ轴旋转
    HWND m_lightXEdit, m_lightYEdit, m_lightZEdit, m_lightIntensityEdit;
    HWND m_light2XEdit, m_light2YEdit, m_light2ZEdit, m_light2IntensityEdit; // 第二个光源
    HWND m_cameraLabel, m_rotationLabel, m_cameraRollLabel, m_lightLabel, m_light2Label;
    
    // 新增：平面光源控件
    HWND m_dirLightXEdit, m_dirLightYEdit, m_dirLightZEdit, m_dirLightIntensityEdit;
    HWND m_dirLightLabel;
    
    // 新增：光照系数控件
    HWND m_diffuseEdit, m_specularEdit, m_ambientEdit;
    HWND m_lightingLabel;
    
    // 新增：高光指数控件
    HWND m_shininessEdit;
    
    // 新增：FOV控制按钮
    HWND m_fovIncreaseBtn, m_fovDecreaseBtn;
    
    // 新增：绘制控制按钮
    HWND m_toggleEdgesBtn, m_toggleRaysBtn;
    
    // 新增：纹理控制按钮
    HWND m_toggleTextureBtn;
    
    // 新增：法线贴图控制按钮
    HWND m_toggleNormalMapBtn;
    
    // 新增：坐标轴和网格线控制按钮
    HWND m_toggleAxesGridBtn;
    
    // 新增：SSAA控制按钮和标签
    HWND m_toggleSSAABtn, m_ssaaScaleIncBtn, m_ssaaScaleDecBtn;
    HWND m_ssaaStatusLabel;
    
    // 新增：模型文件输入控件
    HWND m_modelFileEdit, m_loadModelBtn;
    HWND m_modelStatusLabel;
    
    // 新增：纹理贴图输入控件
    HWND m_textureFileEdit, m_loadTextureBtn;
    HWND m_textureStatusLabel;
    
    // 新增：法线贴图输入控件
    HWND m_normalMapFileEdit, m_loadNormalMapBtn;
    HWND m_normalMapStatusLabel;
    
    // Rendering
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Model> m_model;
    std::shared_ptr<Texture> m_texture;
    std::shared_ptr<Texture> m_normalMap;
    
    // Window properties
    int m_windowWidth, m_windowHeight;
    int m_renderWidth, m_renderHeight;
    
    // Camera settings (摄像机固定在原点，只有角度参数)
    float m_rotationX, m_rotationY, m_rotationZ;
    float m_cameraRollX, m_cameraRollY, m_cameraRollZ; // 摄像机XYZ轴旋转
    
    // Object position settings (物体坐标参数)
    float m_objectX, m_objectY, m_objectZ;
    
    // 新增：FOV设置
    float m_fov;
    
    // Light settings
    float m_lightX, m_lightY, m_lightZ, m_lightIntensity;
    float m_light2X, m_light2Y, m_light2Z, m_light2Intensity; // 第二个光源设置
    
    // 新增：平面光源设置
    float m_dirLightX, m_dirLightY, m_dirLightZ, m_dirLightIntensity;
    
    // 新增：光照系数设置
    Vec3f m_diffuseStrength, m_specularStrength, m_ambientStrength;
    
    // 新增：高光指数设置
    float m_shininess;
    
    // 新增：BRDF 模型参数
    float m_roughness;     // 粗糙度
    float m_metallic;      // 金属度
    
    // 新增：菲涅尔F0参数
    float m_f0R, m_f0G, m_f0B;  // 菲涅尔F0的RGB值
    
    // 新增：能量补偿参数
    float m_energyCompensationScale;  // 能量补偿强度
    
    // 新增：当前模型文件名
    std::string m_currentModelFile;
    
    // 新增：当前纹理文件名
    std::string m_currentTextureFile;
    
    // 新增：当前法线贴图文件名
    std::string m_currentNormalMapFile;
    
    // DIB for displaying
    HBITMAP m_bitmap;
    HDC m_memDC;
    void* m_bitmapData;
    
    // Control IDs
    static const int ID_OBJECT_X = 1001;
    static const int ID_OBJECT_Y = 1002;
    static const int ID_OBJECT_Z = 1003;
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
    
    // 新增：纹理控制ID
    static const int ID_TOGGLE_TEXTURE = 1029;
    
    // 新增：法线贴图控制ID
    static const int ID_TOGGLE_NORMAL_MAP = 1038;
    
    // 新增：坐标轴和网格线控制ID
    static const int ID_TOGGLE_AXES_GRID = 1030;
    
    // 新增：SSAA控制ID
    static const int ID_TOGGLE_SSAA = 1019;
    static const int ID_SSAA_SCALE_INC = 1020;
    static const int ID_SSAA_SCALE_DEC = 1021;
    
    // 新增：第二个光源控制ID
    static const int ID_LIGHT2_X = 1022;
    static const int ID_LIGHT2_Y = 1023;
    static const int ID_LIGHT2_Z = 1024;
    static const int ID_LIGHT2_INTENSITY = 1025;
    
    // 新增：平面光源控制ID
    static const int ID_DIRLIGHT_X = 1048;
    static const int ID_DIRLIGHT_Y = 1049;
    static const int ID_DIRLIGHT_Z = 1050;
    static const int ID_DIRLIGHT_INTENSITY = 1051;
    
    // 新增：光照系数控制ID
    static const int ID_DIFFUSE_STRENGTH = 1026;
    static const int ID_SPECULAR_STRENGTH = 1027;
    static const int ID_AMBIENT_STRENGTH = 1028;
    
    // 新增：高光指数控制ID
    static const int ID_SHININESS = 1031;
    
    // 新增：模型文件控制ID
    static const int ID_MODEL_FILE = 1032;
    static const int ID_LOAD_MODEL = 1033;
    
    // 新增：纹理贴图控制ID
    static const int ID_TEXTURE_FILE = 1034;
    static const int ID_LOAD_TEXTURE = 1035;
    
    // 新增：法线贴图控制ID
    static const int ID_NORMAL_MAP_FILE = 1036;
    static const int ID_LOAD_NORMAL_MAP = 1037;
    
    // 新增：BRDF 控件
    HWND m_roughnessEdit;
    HWND m_metallicEdit;
    HWND m_brdfCheckbox;
    
    // 新增：能量补偿控件
    HWND m_energyCompensationCheckbox;
    HWND m_energyCompensationScaleEdit;
    
    // 新增：菲涅尔F0控件
    HWND m_f0REdit, m_f0GEdit, m_f0BEdit;
    
    // 新增：BRDF 控制ID
    static const int ID_BRDF_ENABLE = 1040;
    static const int ID_ROUGHNESS = 1041;
    static const int ID_METALLIC = 1042;
    
    // 新增：能量补偿控制ID
    static const int ID_ENERGY_COMPENSATION_ENABLE = 1043;
    static const int ID_ENERGY_COMPENSATION_SCALE = 1044;
    
    // 新增：菲涅尔F0控制ID
    static const int ID_F0_R = 1045;
    static const int ID_F0_G = 1046;
    static const int ID_F0_B = 1047;
    
    // 新增：位移着色器控制ID
    static const int ID_TOGGLE_DISPLACEMENT = 1050;
    static const int ID_DISPLACEMENT_SCALE = 1051;
    static const int ID_DISPLACEMENT_FREQ = 1052;
    static const int ID_SPINE_LENGTH = 1053;
    static const int ID_SPINE_SHARPNESS = 1054;
    
    // 新增：平滑移动相关
    void StartMoveTimer();
    void StopMoveTimer();
    void UpdateMovement();
    static const int MOVE_TIMER_ID = 1;
    static const int MOVE_TIMER_INTERVAL = 16; // ~60fps
    bool m_keyStates[256];
    float m_moveSpeed;
    
    // 摄像机平滑移动相关
    Vec3f m_cameraPos;         // 当前摄像机位置
    Vec3f m_cameraTargetPos;   // 目标摄像机位置
    float m_cameraMoveSpeed;   // 摄像机移动速度
    float m_fovTarget;         // 目标FOV
    float m_fovLerpSpeed;      // FOV插值速度
    
    // 摄像机旋转与鼠标拖动
    float m_cameraYaw;
    float m_cameraPitch;
    // 添加四元数相关的变量
    VectorMath::Quaternion m_cameraRotation;
    VectorMath::Quaternion m_targetRotation;
    float m_rotationLerpSpeed;
    bool m_dragging = false;
    POINT m_lastMousePos;
    
    // 添加移动插值相关变量
    Vec3f m_targetPosition;
    float m_moveLerpSpeed;
    bool m_isMoving;
    
    // 新增：自发光控制ID
    static const int ID_TOGGLE_EMISSION = 1052;
    static const int ID_EMISSION_STRENGTH = 1053;
    static const int ID_EMISSION_COLOR_R = 1054;
    static const int ID_EMISSION_COLOR_G = 1055;
    static const int ID_EMISSION_COLOR_B = 1056;
}; 