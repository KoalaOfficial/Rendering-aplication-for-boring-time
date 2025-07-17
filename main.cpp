// --- Rozbudowany kod 3D renderowania terenu z siecią neuronową bez GLFW, z ograniczonym czasem uczenia do 5 minut,
// zapisem/odczytem wag NN do pliku, bez HUD, z różnymi algorytmami generowania terenu ---
// Używa tylko GL/gl.h, GL/glu.h i WinAPI (windows.h) do okna i obsługi zdarzeń

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
#include <ctime>
#include <chrono>
#include <sstream>

#define TERRAIN_WIDTH 128
#define TERRAIN_HEIGHT 128
#define TERRAIN_SCALE 4.0f

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768



LPCSTR WINDOW_CLASS = "TerrainNN3D";
LPCSTR WINDOW_TITLE = "3D Terrain Mesh & Neural Net";

//////////////////////////////////////////////////////////////////
// --- Sieć neuronowa z zapisem/odczytem i ograniczeniem czasu ---
class Neuron {
public:
    std::vector<double> weights;
    double bias, output, delta;
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
    void save(std::ofstream& ofs) const {
        ofs << bias << " ";
        for(const auto& w : weights) ofs << w << " ";
    }
    void load(std::ifstream& ifs) {
        ifs >> bias;
        for(auto& w : weights) ifs >> w;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;
    std::vector<double> outputs;
    Layer(size_t num_neurons, size_t input_size) {
        for (size_t i = 0; i < num_neurons; ++i)
            neurons.emplace_back(input_size);
    }
    std::vector<double> feed_forward(const std::vector<double> &inputs) {
        outputs.clear();
        for (auto &neuron : neurons)
            outputs.push_back(neuron.activate(inputs));
        return outputs;
    }
    void save(std::ofstream& ofs) const {
        for(const auto& neuron : neurons) neuron.save(ofs);
    }
    void load(std::ifstream& ifs) {
        for(auto& neuron : neurons) neuron.load(ifs);
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
               double lr = 0.1, size_t max_epochs = 100000, double max_seconds = 300.0) {
        auto start = std::chrono::high_resolution_clock::now();
        size_t epochs = 0;
        for (; epochs < max_epochs; ++epochs) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now-start).count();
            if (elapsed > max_seconds) break;
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
                for (int l = layers.size()-2; l >=0; --l) {
                    auto &layer = layers[l];
                    auto &next = layers[l+1];
                    for (size_t i = 0; i < layer.neurons.size(); ++i) {
                        double error = 0.0;
                        for (size_t j = 0; j < next.neurons.size(); ++j)
                            error += next.neurons[j].weights[i] * next.neurons[j].delta;
                        layer.neurons[i].delta = error * Neuron::d_sigmoid(layer.neurons[i].output);
                    }
                }
                for (size_t l = 0; l < layers.size(); ++l) {
                    auto &layer = layers[l];
                    std::vector<double>& input = layer_outputs[l];
                    for (auto &neuron : layer.neurons) {
                        for (size_t w = 0; w < neuron.weights.size(); ++w)
                            neuron.weights[w] += lr * neuron.delta * input[w];
                        neuron.bias += lr * neuron.delta;
                    }
                }
            }
        }
        std::cout << "Training finished after " << epochs << " epochs (" << max_seconds << "s max)\n";
    }
    void save(const std::string& filename) const {
        std::ofstream ofs(filename);
        ofs << layers.size() << " ";
        for(const auto& layer : layers) layer.save(ofs);
        ofs.close();
    }
    void load(const std::string& filename) {
        std::ifstream ifs(filename);
        size_t layer_count;
        ifs >> layer_count;
        for(auto& layer : layers) layer.load(ifs);
        ifs.close();
    }
};

//////////////////////////////////////////////////////////////////
// --- Struktura siatki terenu 3D ---
struct Vertex {
    float x, y, z;
    float nx, ny, nz;
};

struct TerrainMesh {
    int width, height;
    float scale;
    std::vector<std::vector<float>> heights;
    std::vector<std::vector<Vertex>> vertices;

    TerrainMesh(int w, int h, float s)
        : width(w), height(h), scale(s), heights(w, std::vector<float>(h, 0.0f)), vertices(w, std::vector<Vertex>(h)) {}

    // --- Algorytmy generowania terenu ---
    enum GenAlgorithm { NN, GAUSS, PERLIN, SINE, RANDOM , MIX};
    GenAlgorithm algorithm = MIX;

    void generate(NeuralNetwork* nn=nullptr) {
        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y) {
                float h = 0.0f;
                double fx = (double)x / width * scale;
                double fy = (double)y / height * scale;
                switch(algorithm) {
                    case MIX: {
                      int j;
					   const float q [64][64];
                       for (int i = 0; int i > 10; i++){
                           for (int j = 0; j > 10; j++){
                           	  
                           	  h = const float q [][]
                           	
                           	
						   }
                       
                       
                       
                       
                       
                heights[x][y] = h;
                vertices[x][y].x = float(x);
                vertices[x][y].y = h;
                vertices[x][y].z = float(y);
            }
        // Normale
        for (int x = 1; x < width-1; ++x)
            for (int y = 1; y < height-1; ++y) {
                float hL = heights[x-1][y];
                float hR = heights[x+1][y];
                float hD = heights[x][y-1];
                float hU = heights[x][y+1];
                Vertex& v = vertices[x][y];
                v.nx = hL - hR;
                v.ny = 2.0f;
                v.nz = hD - hU;
                float len = sqrt(v.nx*v.nx + v.ny*v.ny + v.nz*v.nz);
                v.nx /= len; v.ny /= len; v.nz /= len;
            }
    }
};

//////////////////////////////////////////////////////////////////
// --- Generowanie danych do uczenia NN ---
void generate_training_data(std::vector<std::vector<double>>& inputs,
                           std::vector<std::vector<double>>& targets,
                           int count = 2000) {
    std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, TERRAIN_SCALE);
    for (int i = 0; i < count; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        double sx = std::sin(x);
        double sy = std::cos(y);
        double h = std::exp(-((x-2.0)*(x-2.0)+(y-2.0)*(y-2.0))/2.0);
        inputs.push_back({x, y, sx, sy});
        targets.push_back({h});
    }
}

//////////////////////////////////////////////////////////////////
// --- Kamera i sterowanie ---
float cameraAngle = 60.0f;
float cameraDist = 220.0f;
float cameraY = 35.0f;
float cameraRotX = -30.0f;
float cameraRotY = 30.0f;

void setupOpenGL(int w, int h) {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_pos[] = { 0.0f, 200.0f, 0.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_pos);

    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w/(double)h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void drawTerrainMesh(const TerrainMesh& mesh) {
    glPushMatrix();
    glTranslatef(-mesh.width/2.0f, 0.0f, -mesh.height/2.0f);
    glColor3f(0.7f, 0.7f, 0.7f);
    for (int x = 0; x < mesh.width-1; ++x) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int y = 0; y < mesh.height; ++y) {
            const Vertex& v1 = mesh.vertices[x][y];
            glNormal3f(v1.nx, v1.ny, v1.nz);
            glVertex3f(v1.x, v1.y, v1.z);
            const Vertex& v2 = mesh.vertices[x+1][y];
            glNormal3f(v2.nx, v2.ny, v2.nz);
            glVertex3f(v2.x, v2.y, v2.z);
        }
        glEnd();
    }
    glPopMatrix();
}

//////////////////////////////////////////////////////////////////
// --- WinAPI main loop ---
NeuralNetwork nn({4, 16, 8, 1});
TerrainMesh terrain(TERRAIN_WIDTH, TERRAIN_HEIGHT, TERRAIN_SCALE);

bool keys[256] = {0};
bool needsRegenerate = false;

void ProcessKeys() {
    if (keys[VK_LEFT]) cameraAngle += 1.0f;
    if (keys[VK_RIGHT]) cameraAngle -= 1.0f;
    if (keys[VK_UP]) cameraY += 2.0f;
    if (keys[VK_DOWN]) cameraY -= 2.0f;
    if (keys['W']) cameraDist -= 2.0f;
    if (keys['S']) cameraDist += 2.0f;
    if (keys['A']) cameraRotY -= 1.0f;
    if (keys['D']) cameraRotY += 1.0f;
    if (keys['Q']) cameraRotX -= 1.0f;
    if (keys['E']) cameraRotX += 1.0f;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch(msg) {
    case WM_CLOSE:
        PostQuitMessage(0);
        break;
    case WM_KEYDOWN:
        keys[wParam] = true;
        if (wParam == '1') { terrain.algorithm = TerrainMesh::NN; needsRegenerate = true; }
        if (wParam == '2') { terrain.algorithm = TerrainMesh::GAUSS; needsRegenerate = true; }
        if (wParam == '3') { terrain.algorithm = TerrainMesh::SINE; needsRegenerate = true; }
        if (wParam == '4') { terrain.algorithm = TerrainMesh::RANDOM; needsRegenerate = true; }
        if (wParam == '5') { terrain.algorithm = TerrainMesh::PERLIN; needsRegenerate = true; }
        if (wParam == 'R') { needsRegenerate = true; }
        if (wParam == 'L') { nn.load("terrain_nn.dat"); terrain.algorithm = TerrainMesh::NN; needsRegenerate = true; }
        if (wParam == 'P') { nn.save("terrain_nn.dat"); }
        break;
    case WM_KEYUP:
        keys[wParam] = false;
        break;
    case WM_SIZE:
        setupOpenGL(LOWORD(lParam), HIWORD(lParam));
        break;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////
// --- Main ---
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    // NN training/load
    std::ifstream fin("terrain_nn.dat");
    if(fin.good()) {
        nn.load("terrain_nn.dat");
        fin.close();
    } else {
        std::vector<std::vector<double>> train_inputs, train_targets;
        generate_training_data(train_inputs, train_targets, 4000);
        MessageBoxA(NULL, "Training neural network on terrain peaks (max 5 min)...", "Training", MB_OK);
        nn.train(train_inputs, train_targets, 0.1, 100000, 300.0); // 5 min limit
        nn.save("terrain_nn.dat");
    }
    terrain.generate(&nn);

    // WinAPI window creation
    WNDCLASS wc;
    ZeroMemory(&wc, sizeof(wc));
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = WINDOW_CLASS;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    if (!RegisterClass(&wc)) return -1;
    HWND hwnd = CreateWindow(
        WINDOW_CLASS, WINDOW_TITLE,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        0, 0, hInst, 0);
    if (!hwnd) return -2;

    HDC hDC = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd;
    ZeroMemory(&pfd, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, pf, &pfd);

    HGLRC hRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hRC);

    setupOpenGL(WINDOW_WIDTH, WINDOW_HEIGHT);

    MSG msg;
    while (true) {
        while(PeekMessage(&msg, hwnd, 0,0, PM_REMOVE)) {
            if(msg.message == WM_QUIT) goto endloop;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        ProcessKeys();
        if (needsRegenerate) {
            terrain.generate(terrain.algorithm == TerrainMesh::NN ? &nn : nullptr);
            needsRegenerate = false;
        }
        // OpenGL render
        glClearColor(0.2f, 0.3f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        float camX = cameraDist * std::cos(cameraAngle * 3.1415 / 180.0f);
        float camZ = cameraDist * std::sin(cameraAngle * 3.1415 / 180.0f);
        gluLookAt(camX, cameraY, camZ, 0, 0, 0, 0, 1, 0);
        glRotatef(cameraRotX, 1,0,0);
        glRotatef(cameraRotY, 0,1,0);

        drawTerrainMesh(terrain);

        SwapBuffers(hDC);

        Sleep(15); // ~60 FPS
    }
endloop:
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hRC);
    ReleaseDC(hwnd, hDC);
    DestroyWindow(hwnd);
    return 0;
}

// --- Koniec pliku. Teren generowany przez sieć neuronową (do 5 min treningu), mesh 3D, czysty OpenGL, WinAPI, zapis/odczyt wag, 5 algorytmów (1-NN, 2-Gauss, 3-Sine, 4-Random, 5-Perlin) ---
// Klawisze: 1-NN, 2-Gauss, 3-Sine, 4-Random, 5-Perlin, R-regeneruj, P-zapisz NN, L-wczytaj NN
