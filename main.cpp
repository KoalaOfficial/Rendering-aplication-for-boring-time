#include <windows.h>
#include <gl/gl.h>
#include <cmath>
#include <gl/glu.h>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

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

// --- PARAMETRY TERENU ---
#define TERRAIN_WIDTH 128
#define TERRAIN_HEIGHT 128

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
    unsigned char perm[512];
    void InitPerm() {
        std::vector<unsigned char> p(256);
        for (int i = 0; i < 256; ++i) p[i] = i;
        std::mt19937 gen(seed);
        std::shuffle(p.begin(), p.end(), gen);
        for (int i = 0; i < 512; ++i)
            perm[i] = p[i % 256];
    }

    void GenerateTerrain() {
        InitPerm();
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                float h = 0.0f;
                for (auto &layer : layers) {
                    float freq = layer.frequency;
                    float amp = layer.amplitude;
                    for (int o = 0; o < layer.octaves; ++o) {
                        float fx = freq * (float)x / TERRAIN_WIDTH;
                        float fy = freq * (float)y / TERRAIN_HEIGHT;
                        h += Perlin(fx, fy) * amp;
                        freq *= 2.0f;
                        amp *= 0.5f;
                    }
                }
                heightMap[y][x] = h;
            }
        }
    }

    void GenerateMountains() {
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                float dist = std::sqrt(
                    std::pow((x - TERRAIN_WIDTH/2.0f)/TERRAIN_WIDTH, 2) +
                    std::pow((y - TERRAIN_HEIGHT/2.0f)/TERRAIN_HEIGHT, 2));
                heightMap[y][x] = std::exp(-dist*16) * 3.0f + heightMap[y][x] * 0.5f;
            }
        }
    }

    void GenerateLake() {
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                float dx = (x - TERRAIN_WIDTH/2.0f) / (TERRAIN_WIDTH/2.0f);
                float dy = (y - TERRAIN_HEIGHT/2.0f) / (TERRAIN_HEIGHT/2.0f);
                float dist = std::sqrt(dx*dx + dy*dy);
                if (dist < 0.4)
                    heightMap[y][x] = -2.0f + 2.0f * dist;
            }
        }
    }

    void GenerateForest() {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> tree_dist(0.5f, 1.2f);
        for (int y = 0; y < TERRAIN_HEIGHT; ++y) {
            for (int x = 0; x < TERRAIN_WIDTH; ++x) {
                if ((x+y)%11 == 0)
                    heightMap[y][x] += tree_dist(gen);
            }
        }
    }

    void GeneratePlains() {
        for (int y = 0; y < TERRAIN_HEIGHT; ++y)
            for (int x = 0; x < TERRAIN_WIDTH; ++x)
                heightMap[y][x] = 0.0f;
    }
};

// --- KAMERA GRACZA ---
struct Camera {
    float x, y, z;
    float pitch, yaw;
    float moveSpeed, turnSpeed;

    Camera() : x(TERRAIN_WIDTH/2.0f), z(TERRAIN_HEIGHT/2.0f), y(10.0f), pitch(0), yaw(0), moveSpeed(1.5f), turnSpeed(0.03f) {}

    void Move(float forward, float strafe, TerrainGenerator& terrain) {
        float rad = yaw * M_PI / 180.0f;
        float dx = std::cos(rad) * forward - std::sin(rad) * strafe;
        float dz = std::sin(rad) * forward + std::cos(rad) * strafe;
        x += dx * moveSpeed;
        z += dz * moveSpeed;
        if (x < 0) x = 0; if (x > TERRAIN_WIDTH-1) x = TERRAIN_WIDTH-1;
        if (z < 0) z = 0; if (z > TERRAIN_HEIGHT-1) z = TERRAIN_HEIGHT-1;
        y = terrain.heightMap[(int)z][(int)x] + 3.0f; // wysokość nad terenem
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
void RenderTerrain(TerrainGenerator& terrain, Camera& camera, HDC hDC) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, 800, 600);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(70.0, 800.0/600.0, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float camY = camera.y;
    float lookX = camera.x + std::cos(camera.yaw * M_PI / 180.0f);
    float lookZ = camera.z + std::sin(camera.yaw * M_PI / 180.0f);
    gluLookAt(camera.x, camY, camera.z, lookX, camY + std::tan(camera.pitch * M_PI / 180.0f), lookZ, 0, 1, 0);

    // Renderowanie siatki terenu
    glColor3f(0.3f, 0.8f, 0.3f);
    for (int y = 0; y < TERRAIN_HEIGHT-1; ++y) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int x = 0; x < TERRAIN_WIDTH; ++x) {
            glVertex3f(x, terrain.heightMap[y][x], y);
            glVertex3f(x, terrain.heightMap[y+1][x], y+1);
        }
        glEnd();
    }
    SwapBuffers(hDC);
}

// --- MAIN ---
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Inicjalizacja generatora terenu
    TerrainGenerator terrain;

    // Sieć neuronowa (4 klasy krajobrazu)
    NeuralNetwork nn({16, 8, 4});
    std::vector<std::vector<double>> train_inputs = {
        {0.9,0.8,0.7,0.5,0.2,0.3,0.1,0.0,0.2,0.1,0.3,0.2,0.1,0.0,0.2,0.1},  // góry
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

    terrain.GenerateTerrain();
    if (terrain_type == "gory") terrain.GenerateMountains();
    else if (terrain_type == "jezioro") terrain.GenerateLake();
    else if (terrain_type == "las") terrain.GenerateForest();
    else if (terrain_type == "rownina") terrain.GeneratePlains();

    // Wygładzanie terenu
    for (int it = 0; it < 2; ++it) {
        float temp[TERRAIN_HEIGHT][TERRAIN_WIDTH];
        for (int y = 1; y < TERRAIN_HEIGHT-1; ++y) {
            for (int x = 1; x < TERRAIN_WIDTH-1; ++x) {
                float sum = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        sum += terrain.heightMap[y+dy][x+dx];
                temp[y][x] = sum / 9.0f;
            }
        }
        for (int y = 1; y < TERRAIN_HEIGHT-1; ++y)
            for (int x = 1; x < TERRAIN_WIDTH-1; ++x)
                terrain.heightMap[y][x] = temp[y][x];
    }

    // --- Tworzenie okna i kontekstu OpenGL ---
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "TerrainWindow";
    RegisterClass(&wc);

    HWND hWnd = CreateWindow(
        wc.lpszClassName,
        "Procedural Terrain 3D",
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

    // --- Kamera gracza ---
    Camera camera;

    // --- Główna pętla renderowania i sterowania ---
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
     
