#ifndef RENDERER_H
#define RENDERER_H

#include <windows.h>
#include <gl/gl.h>
#include "terrain.h"

class TerrainRenderer {
private:
    TerrainGenerator* terrain;
    
    // Parametry kamery
    float cameraX, cameraY, cameraZ;
    float cameraYaw, cameraPitch;
    float cameraYawMin, cameraYawMax;
    float cameraPitchMin, cameraPitchMax;
    
    // Parametry renderowania
    int width, height;
    bool showLoadingScreen;
    float loadingProgress;
    
    // Tekstury i materiały
    GLuint groundTexture;
    GLuint skyboxTexture;
    
    void SetupLighting();
    void SetViewMatrix();
    void SetProjection(int width, int height);
    void DrawLoadingScreen();

public:
    TerrainRenderer(TerrainGenerator* t);
    ~TerrainRenderer();
    
    void Initialize(HWND hWnd);
    void Render(HDC hDC);
    void HandleInput();
    void Resize(int w, int h);
    
    // Nowe funkcje
    void EnableLoadingScreen(bool enable);
    void SetLoadingProgress(float progress);
    void LoadTextures();
    void SetupSkybox();
};

#endif
