#include <windows.h>
#include <gl/gl.h>
#include <gl/glu.h>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

// --- PARAMETRY TERENU ---
#define TERRAIN_WIDTH 256
#define TERRAIN_HEIGHT 256

enum TerrainType { MOUNTAIN, LAKE, FOREST, PLAIN, HILL, VALLEY, CLIFF, ISLAND, RIVER, BEACH };

TerrainType biomeMap[TERRAIN_HEIGHT][TERRAIN_WIDTH];

// --- SIEĆ NEURONOWA ---
class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;
    Neuron(size_t input_size) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        weights.resize(input_size);
        for (auto &w : weights) w = dist(gen);
        bias = dist(gen);
    }
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double d_sigmoid(double x) { return x * (1.0 - x); }
    double activate(const std::vector<double> &inputs) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i)
            sum += weights[i] * inputs[i];
        output = sigmoid(sum);
        return output;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;
    Layer(size_t num_neurons, size_t input_size) {
        for (size_t i = 0; i < num_neurons; ++i)
            neurons.emplace_back(input_size);
    }
    std::vector<double> feed_forward(const std::vector<double> &inputs) {
        std::vector<double> outputs;
        for (auto &neuron : neurons)
            outputs.push_back(neuron.activate(inputs));
        return outputs;
    }
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    NeuralNetwork(const std::vector<size_t> &layer_sizes) {
        assert(layer_sizes.size() >= 2);
        for (size_t i = 1; i < layer_sizes.size(); ++i)
            layers.emplace_back(layer_sizes[i], layer_sizes[i-1]);
    }
    std::vector<double> predict(const std::vector<double> &input) {
        std::vector<double> out = input;
        for (auto &layer : layers)
            out = layer.feed_forward(out);
        return out;
    }
    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets,
               double lr = 0.1, size_t epochs = 1000) {
        for (size_t e = 0; e < epochs; ++e) {
            for (size_t sample = 0; sample < inputs.size(); ++sample) {
                std::vector<std::vector<double>> layer_outputs;
                layer_outputs.push_back(inputs[sample]);
                for (auto &layer : layers)
                    layer_outputs.push_back(layer.feed_forward(layer_outputs.back()));
                auto &out_layer = layers.back();
                for (size_t i = 0; i < out_layer.neurons.size(); ++i) {
                    double error = targets[sample][i] - out_layer.neurons[i].output;
                    out_layer.neurons[i].delta = error * Neuron::d_sigmoid(out_layer.neurons[i].output);
                }
                for (int l = layers.size() - 2; l >= 0; --l) {
                    auto &layer = layers[l];
                    auto &next_layer = layers[l+1];
                    for (size_t i = 0; i < layer.neurons.size(); ++i) {
                        double error = 0.0;
                        for (auto &next_neuron : next_layer.neurons)
                            error += next_neuron.weights[i] * next_neuron.delta;
                        layer.neurons[i].delta = error * Neuron::d_sigmoid(layer.neurons[i].output);
                    }
                }
                for (size_t l = 0; l < layers.size(); ++l) {
                    auto &layer = layers[l];
                    auto &inputs_ = layer_outputs[l];
                    for (auto &neuron : layer.neurons) {
                        for (size_t w = 0; w < neuron.weights.size(); ++w)
                            neuron.weights[w] += lr * neuron.delta * inputs_[w];
                        neuron.bias += lr * neuron.delta;
                    }
                }
            }
        }
    }
};

// --- GENERATOR TERENU ---
struct TerrainLayer {
    float frequency;
    float amplitude;
    int octaves;
};

class TerrainGenerator {
public:
    float heightMap[TERRAIN_HEIGHT][TERRAIN_WIDTH];
    std::vector<TerrainLayer> layers;
    unsigned int seed = 42;
    unsigned char perm[512];

    TerrainGenerator() {
        layers.push_back({8.0f, 1.0f, 4});
        layers.push_back({16.0f, 0.5f, 3});
    }

    float Fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    float Lerp(float a, float b, float t) { return a + t * (b - a); }
    float Grad(int hash, float x, float y) {
        int h = hash & 3;
        float u = h < 2 ? x : y;
        float v = h < 2 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
    }
    float Perlin(float x, float y) {
        int xi = (int)x & 255, yi = (int)y & 255;
        float xf = x - (int)x, yf = y - (int)y;
        float u = Fade(xf), v = Fade(yf);

        int aa = perm[xi] + yi, ab = perm[xi] + yi + 1;
        int ba = perm[xi + 1] + yi, bb = perm[xi + 1] + yi + 1;

        float x1 = Lerp(Grad(perm[aa], xf, yf), Grad(perm[ba], xf - 1, yf), u);
        float x2 = Lerp(Grad(perm[ab], xf, yf - 1), Grad(perm[bb], xf - 1, yf - 1), u);
        return (Lerp(x1, x2, v) + 1.0f) / 2.0f;
    }
    void InitPerm() {
        std::vector<unsigned char> p(256);
        for (int i = 0; i < 256; ++i) p[i] = i;
        std::mt19937 gen(seed);
        std::shuffle(p.begin(), p.end(), gen);
        for (int i = 0; i < 512; ++i)
            perm[i] = p[i % 256];
    }

    void GenerateBiomeMap() {
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                float fx = (float)x / TERRAIN_WIDTH;
                float fy = (float)y / TERRAIN_HEIGHT;
                float noise = Perlin(fx * 4.0f, fy * 4.0f);

                if (noise > 0.75f) biomeMap[y][x] = MOUNTAIN;
                else if (noise < 0.18f) biomeMap[y][x] = LAKE;
                else if (noise < 0.25f) biomeMap[y][x] = RIVER;
                else if (noise < 0.32f) biomeMap[y][x] = VALLEY;
                else if (noise < 0.5f) biomeMap[y][x] = FOREST;
                else if (noise < 0.60f) biomeMap[y][x] = HILL;
                else if (noise < 0.68f) biomeMap[y][x] = CLIFF;
                else biomeMap[y][x] = PLAIN;

                // Wyspy w losowych miejscach (jako przykład)
                if ((x-128)*(x-128)+(y-128)*(y-128) < 1800 && Perlin(fx*6.5f, fy*6.5f)>0.6f)
                    biomeMap[y][x] = ISLAND;

                // Plaża przy jeziorach
                if (biomeMap[y][x] == LAKE && Perlin(fx*15.0f, fy*15.0f) > 0.45f)
                    biomeMap[y][x] = BEACH;
            }
        }
    }

    void GenerateTerrainWithBiomes() {
        InitPerm();
        GenerateBiomeMap();
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                switch (biomeMap[y][x]) {
                    case MOUNTAIN:
                        heightMap[y][x] = Perlin(x * 0.12f, y * 0.12f) * 12 + 11 * std::exp(-0.025 * ((x - TERRAIN_WIDTH/2)*(x - TERRAIN_WIDTH/2) + (y - TERRAIN_HEIGHT/2)*(y - TERRAIN_HEIGHT/2)));
                        break;
                    case LAKE:
                        heightMap[y][x] = -3.0f + Perlin(x * 0.4f, y * 0.4f) * 0.5f;
                        break;
                    case FOREST:
                        heightMap[y][x] = 2.0f + Perlin(x * 0.16f, y * 0.16f) * 2.5f;
                        break;
                    case HILL:
                        heightMap[y][x] = 3.3f + Perlin(x * 0.2f, y * 0.2f) * 3.0f + std::sin(x * 0.08f) * std::cos(y * 0.09f);
                        break;
                    case VALLEY:
                        heightMap[y][x] = -2.0f + Perlin(x * 0.33f, y * 0.33f) * 1.1f;
                        break;
                    case CLIFF:
                        heightMap[y][x] = ((x % 22 < 6) ? 13.0f : Perlin(x * 0.2f, y * 0.2f) * 2.2f);
                        break;
                    case ISLAND: {
                        float dx = (x - TERRAIN_WIDTH/2.0f) / (TERRAIN_WIDTH/2.0f);
                        float dy = (y - TERRAIN_HEIGHT/2.0f) / (TERRAIN_HEIGHT/2.0f);
                        float dist = std::sqrt(dx*dx + dy*dy);
                        heightMap[y][x] = (dist < 0.7 ? 5.0f - 8.0f * dist : -3.0f) + Perlin(x * 0.25f, y * 0.25f);
                        break;
                    }
                    case RIVER:
                        heightMap[y][x] = -4.0f + Perlin(x * 0.6f, y * 0.12f) * 0.5f;
                        break;
                    case BEACH:
                        heightMap[y][x] = -1.2f + Perlin(x * 0.2f, y * 0.2f) * 0.5f;
                        break;
                    case PLAIN:
                    default:
                        heightMap[y][x] = 1.0f + Perlin(x * 0.07f, y * 0.07f) + std::sin(x * 0.09f) * std::cos(y * 0.16f) * 0.7f;
                        break;
                }
            }
        }
    }

    void SmoothBiome(TerrainType biome, int iterations = 2) {
        for (int it = 0; it < iterations; ++it) {
            float temp[TERRAIN_HEIGHT][TERRAIN_WIDTH] = {};
            for (int y = 1; y < TERRAIN_HEIGHT-1; ++y) {
                for (int x = 1; x < TERRAIN_WIDTH-1; ++x) {
                    if (biomeMap[y][x] == biome) {
                        float sum = 0.0f;
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                sum += heightMap[y+dy][x+dx];
                        temp[y][x] = sum / 9.0f;
                    }
                }
            }
            for (int y = 1; y < TERRAIN_HEIGHT-1; ++y)
                for (int x = 1; x < TERRAIN_WIDTH-1; ++x)
                    if (biomeMap[y][x] == biome)
                        heightMap[y][x] = temp[y][x];
        }
    }

    // Dodaj rzekę przechodzącą przez mapę
    void GenerateRiver() {
        std::mt19937 gen(seed);
        int x = gen() % TERRAIN_WIDTH;
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int w = -2; w <= 2; ++w) {
                int rx = x + w;
                if (rx >= 0 && rx < TERRAIN_WIDTH) {
                    biomeMap[y][rx] = RIVER;
                    heightMap[y][rx] = -4.5f + Perlin(rx * 0.1f, y * 0.1f);
                }
            }
            x += (gen() % 3 - 1);
            if (x < 0) x = 0;
            if (x >= TERRAIN_WIDTH) x = TERRAIN_WIDTH-1;
        }
    }
};

// --- KAMERA GRACZA ---
struct Camera {
    float x, y, z;
    float pitch, yaw;
    float moveSpeed, turnSpeed;
    Camera() : x(TERRAIN_WIDTH/2.0f), z(TERRAIN_HEIGHT/2.0f), y(16.0f), pitch(0), yaw(0), moveSpeed(2.6f), turnSpeed(0.04f) {}
    void Move(float forward, float strafe, TerrainGenerator& terrain) {
        float rad = yaw * M_PI / 180.0f;
        float dx = std::cos(rad) * forward - std::sin(rad) * strafe;
        float dz = std::sin(rad) * forward + std::cos(rad) * strafe;
        x += dx * moveSpeed;
        z += dz * moveSpeed;
        if (x < 0) x = 0; if (x > TERRAIN_WIDTH-1) x = TERRAIN_WIDTH-1;
        if (z < 0) z = 0; if (z > TERRAIN_HEIGHT-1) z = TERRAIN_HEIGHT-1;
        y = terrain.heightMap[(int)z][(int)x] + 4.0f;
    }
    void Turn(float dpitch, float dyaw) {
        pitch += dpitch * turnSpeed;
        yaw += dyaw * turnSpeed;
        if (pitch < -80) pitch = -80;
        if (pitch > 80) pitch = 80;
        if (yaw < 0) yaw += 360;
        if (yaw > 360) yaw -= 360;
    }
};

// --- KLASYFIKACJA KRAJOBRAZU ---
std::string recognize_landscape(const std::vector<double>& image_features, NeuralNetwork& nn) {
    auto result = nn.predict(image_features);
    int idx = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
    switch (idx) {
        case 0: return "gory";
        case 1: return "jezioro";
        case 2: return "las";
        case 3: return "rownina";
        default: return "nieznane";
    }
}

// --- RENDERER OPENGL ---
// Kolory dla biomów:
void setBiomeColor(TerrainType biome) {
    switch (biome) {
        case MOUNTAIN: glColor3f(0.5f, 0.4f, 0.4f); break;
        case LAKE:     glColor3f(0.2f, 0.4f, 0.85f); break;
        case FOREST:   glColor3f(0.1f, 0.6f, 0.2f); break;
        case PLAIN:    glColor3f(0.7f, 0.8f, 0.5f); break;
        case HILL:     glColor3f(0.6f, 0.7f, 0.3f); break;
        case VALLEY:   glColor3f(0.2f, 0.5f, 0.3f); break;
        case CLIFF:    glColor3f(0.7f, 0.5f, 0.2f); break;
        case ISLAND:   glColor3f(0.85f, 0.85f, 0.4f); break;
        case RIVER:    glColor3f(0.0f, 0.4f, 0.8f); break;
        case BEACH:    glColor3f(0.95f, 0.92f, 0.65f); break;
        default:       glColor3f(0.4f, 0.4f, 0.4f);
    }
}

void RenderTerrain(TerrainGenerator& terrain, Camera& camera, HDC hDC) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, 800, 600);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(70.0, 800.0/600.0, 0.1, 1300.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float camY = camera.y;
    float lookX = camera.x + std::cos(camera.yaw * M_PI / 180.0f);
    float lookZ = camera.z + std::sin(camera.yaw * M_PI / 180.0f);
    gluLookAt(camera.x, camY, camera.z, lookX, camY + std::tan(camera.pitch * M_PI / 180.0f), lookZ, 0, 1, 0);

    // Renderowanie terenu z kolorami biomów
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (int y = 0; y < TERRAIN_HEIGHT-1; ++y) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int x = 0; x < TERRAIN_WIDTH; ++x) {
            setBiomeColor(biomeMap[y][x]);
            glVertex3f(x, terrain.heightMap[y][x], y);
            setBiomeColor(biomeMap[y+1][x]);
            glVertex3f(x, terrain.heightMap[y+1][x], y+1);
        }
        glEnd();
    }
    SwapBuffers(hDC);
}

// --- MAIN ---
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    TerrainGenerator terrain;

    NeuralNetwork nn({16, 8, 4});
    std::vector<std::vector<double>> train_inputs = {
        {6.9,0.8,0.7,0.5,0.2,0.3,0.1,0.0,0.2,0.1,0.3,0.2,0.1,0.0,0.2,9.6},  // góry
        {0.1,0.2,0.3,0.4,0.8,0.9,0.7,0.6,0.8,0.7,0.9,0.8,0.7,0.6,0.8,0.7},  // jezioro
        {0.7,0.6,0.5,0.8,0.9,0.8,0.7,0.6,0.7,0.6,0.9,0.8,0.7,0.6,0.7,0.6},  // las
        {0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3}   // równina
    };
    std::vector<std::vector<double>> train_targets = {
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}
    };
    nn.train(train_inputs, train_targets, 0.2, 500);

    std::vector<double> image_features = {0.9,0.7,0.6,0.5,0.2,0.2,0.1,0.1,0.2,0.1,0.3,0.2,0.1,0.0,0.2,0.1};
    std::string terrain_type = recognize_landscape(image_features, nn);

    terrain.GenerateTerrainWithBiomes();
    terrain.GenerateRiver();
    terrain.SmoothBiome(MOUNTAIN, 2);
    terrain.SmoothBiome(HILL, 1);

    // --- Tworzenie okna i kontekstu OpenGL ---
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "TerrainWindow";
    RegisterClass(&wc);

    HWND hWnd = CreateWindow(
        wc.lpszClassName,
        "Procedural Terrain 3D - Advanced",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        800, 600,
        NULL, NULL,
        hInstance, NULL
    );
    HDC hDC = GetDC(hWnd);

    PIXELFORMATDESCRIPTOR pfd = {sizeof(PIXELFORMATDESCRIPTOR), 1};
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    int pf = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, pf, &pfd);
    HGLRC hGLRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hGLRC);

    Camera camera;

    MSG msg;
    bool running = true;
    while (running) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        // Sterowanie klawiaturą (WASD, strzałki)
        if (GetAsyncKeyState('W')) camera.Move(1, 0, terrain);
        if (GetAsyncKeyState('S')) camera.Move(-1, 0, terrain);
        if (GetAsyncKeyState('A')) camera.Move(0, -1, terrain);
        if (GetAsyncKeyState('D')) camera.Move(0, 1, terrain);
        if (GetAsyncKeyState(VK_UP)) camera.Turn(1, 0);
        if (GetAsyncKeyState(VK_DOWN)) camera.Turn(-1, 0);
        if (GetAsyncKeyState(VK_LEFT)) camera.Turn(0, -2);
        if (GetAsyncKeyState(VK_RIGHT)) camera.Turn(0, 2);

        RenderTerrain(terrain, camera, hDC);
        Sleep(16); // ~60 FPS
    }

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hGLRC);
    ReleaseDC(hWnd, hDC);
    DestroyWindow(hWnd);

    return 0;
}
