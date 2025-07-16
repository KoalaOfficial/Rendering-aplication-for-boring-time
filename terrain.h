#ifndef TERRAIN_H
#define TERRAIN_H

#include <windows.h>
#include <gl/gl.h>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>

#define M_PI 3.14159265358979323846

class TerrainGenerator {
private:
    // Parametry terenu
    static const int TERRAIN_WIDTH = 256;
    static const int TERRAIN_HEIGHT = 256;
    
    float* heightMap;
    float normals[256][256][3];
    unsigned char* perlinPermutation;
    float* perlinGradient;
    
    // Parametry generowania
    int fractalDepth;
    float fractalAmplitude;
    unsigned int seed;
    float frequency;
    
    // Warstwy terenu
    struct TerrainLayer {
        float frequency;
        float amplitude;
        int octaves;
    };
    std::vector<TerrainLayer> layers;

    // Metody pomocnicze
    float Fade(float t);
    float Lerp(float a, float b, float t);
    float Gradient(int hash, float x, float z);
    void GeneratePerlinGradient();
    void InitializePermutation();
    float PerlinNoise(float x, float z);
    float FractalNoise(float x, float z, int depth, float amplitude);
    void CalculateNormalAt(int x, int z, float* normal);

public:
    TerrainGenerator();
    ~TerrainGenerator();
    
    // Główne metody
    void GenerateTerrain();
    void SmoothTerrain(int iterations);
    void CalculateNormals();
    float GetHeightAt(int x, int z);
    
    // Nowe funkcjonalności
    void AddLayer(float freq, float amp, int oct);
    bool SaveToFile(const char* filename);
    bool LoadFromFile(const char* filename);
    void RenderLoadingScreen(HDC hDC, float progress);
    
    // Gettery dla renderera
    const float* GetHeightMap() const { return heightMap; }
    const float* GetNormalAt(int x, int z) const { return normals[x][z]; }
    int GetWidth() const { return TERRAIN_WIDTH; }
    int GetHeight() const { return TERRAIN_HEIGHT; }
};

#endif
