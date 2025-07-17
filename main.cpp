// main.cpp

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

#define TERRAIN_WIDTH 256
#define TERRAIN_HEIGHT 256
#define TERRAIN_SCALE 5.0f

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

LPCSTR WINDOW_CLASS = "TerrainNN3D";
LPCSTR WINDOW_TITLE = "3D Terrain Mesh & Neural Net";

// --- Sieć neuronowa ---
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
               double lr = 0.1, size_t max_epochs = 100000, double max_seconds = 120.0) {
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

// --- Perlin noise helpers ---
float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
float lerp(float a, float b, float t) { return a + t * (b - a); }
float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}
float perlin2d(float x, float y, const std::vector<int>& p) {
    int xi = int(x) & 255;
    int yi = int(y) & 255;

    float xf = x - int(x);
    float yf = y - int(y);

    float u = fade(xf);
    float v = fade(yf);

    int aa = p[p[xi] + yi];
    int ab = p[p[xi] + yi + 1];
    int ba = p[p[xi + 1] + yi];
    int bb = p[p[xi + 1] + yi + 1];

    float x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u);
    float x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u);

    return lerp(x1, x2, v);
}
float fractal_perlin(float x, float y, const std::vector<int>& p, int octaves = 5, float persistence = 0.5f) {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        total += perlin2d(x * frequency, y * frequency, p) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }
    return total / maxValue;
}
float gradient(float fx, float fy, float scale = 1.0f) {
    return (fx + fy) * scale;
}

// --- Realistyczne generatory terenów ---
float mountain_func(float x, float y) {
    // Szczyty gór, strome zbocza + ridge noise
    float r = std::sqrt((x-1.5)*(x-1.5)+(y-1.5)*(y-1.5));
    float peak = std::exp(-(r*r)*0.8)*35.0f + std::sin(x*4 + y*3)*8.0f;
    float ridge = std::abs(std::sin(x*7 + y*6.5)*10.0f);
    return peak + ridge - 13f;
}
float valley_func(float x, float y) {
    // Szerokie doliny, łagodne zbocza
    float r = std::sqrt((x-2.8)*(x-2.8)+(y-0.5)*(y-0.5));
    return -std::exp(-r * 1.3) * 30.0f + std::sin(x*2.0 + y*2.0)*2.0f;
}
float plateau_func(float x, float y) {
    // Płaskowyż
    float base = std::exp(-((x-2.2)*(x-2.2)+(y-2.2)*(y-2.2))*1.2)*22.0f;
    float flat = std::max(0.0f, float(1.0 - std::abs(x-2.2)*0.7 - std::abs(y-2.2)*0.7))*5.0f;
    return base + flat + std::sin(x*1.5 + y*1.5)*2.5f;
}
float canyon_func(float x, float y) {
    // Kanion: głębokie, wąskie obniżenie
    float c = std::sin(x*7.5)*std::cos(y*3.5)*10.0f - std::abs(x-1.7)*18.0f;
    float noise = std::sin(x*2.0+y*2.5)*2.0f;
    return c + noise;
}
float hill_func(float x, float y) {
    // Pagórki
    return std::sin(x*5.0 + y*3.0)*12.0f + std::sin(x*6.0)*4.0f + std::cos(y*6.0)*4.0f;
}
float plain_func(float x, float y) {
    // Równina z lekką falą
    return std::sin(x*1.1)*2.0f + std::sin(y*1.3)*2.0f;
}


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
    enum GenAlgorithm {
        NN, GAUSS, PERLIN, SINE, RANDOM,
        FRACTAL, GRADIENT, MIX,
        MOUNTAIN, VALLEY, PLATEAU, CANYON, HILL, PLAIN
    };
    GenAlgorithm algorithm = MIX;

    std::vector<int> perlin_perm;
    void initPerlin() {
        perlin_perm.resize(512);
        std::vector<int> p(256);
        for (int i = 0; i < 256; ++i) p[i] = i;
        std::mt19937 gen(1337);
        std::shuffle(p.begin(), p.end(), gen);
        for (int i = 0; i < 512; ++i) perlin_perm[i] = p[i & 255];
    }

    void generate(NeuralNetwork* nn = nullptr) {
        if (algorithm == PERLIN || algorithm == FRACTAL || algorithm == MIX ||
            algorithm == MOUNTAIN || algorithm == VALLEY || algorithm == CANYON || algorithm == HILL) {
            if (perlin_perm.empty()) initPerlin();
        }
        static std::mt19937 gen(time(0));
        static std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y) {
                float h = 0.0f;
                double fx = (double)x / width * scale;
                double fy = (double)y / height * scale;

                switch (algorithm) {
                    case NN: {
                        if (nn) {
                            std::vector<double> inp = { fx, fy, std::sin(fx), std::cos(fy) };
                            auto out = nn->predict(inp);
                            h = float(out[0] * 40.0f - 20.0f);
                        }
                        break;
                    }
                    case GAUSS: {
                        h = float(std::exp(-((fx - 2.0) * (fx - 2.0) + (fy - 2.0) * (fy - 2.0)) / 2.0) * 40.0f - 20.0f);
                        break;
                    }
                    case SINE: {
                        h = float((std::sin(fx * 2.0) + std::cos(fy * 2.0)) * 10.0f);
                        break;
                    }
                    case RANDOM: {
                        h = dist(gen);
                        break;
                    }
                    case PERLIN: {
                        h = fractal_perlin(fx, fy, perlin_perm, 6, 0.5f) * 25.0f;
                        break;
                    }
                    case FRACTAL: {
                        float sum = 0.0f, amp = 10.0f;
                        for (int o = 1; o <= 6; ++o) {
                            sum += std::sin(fx * o + fy * o) * amp;
                            amp *= 0.5f;
                        }
                        sum += fractal_perlin(fx, fy, perlin_perm, 5, 0.6f) * 10.0f;
                        h = sum;
                        break;
                    }
                    case GRADIENT: {
                        h = gradient(fx, fy, 7.0f);
                        break;
                    }
                    case MOUNTAIN: {
                        h = mountain_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 3, 0.7f) * 8.0f;
                        break;
                    }
                    case VALLEY: {
                        h = valley_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 2, 0.6f) * 6.0f;
                        break;
                    }
                    case PLATEAU: {
                        h = plateau_func(fx, fy) + std::sin(fx*3.5+fy*3.5)*2.0f;
                        break;
                    }
                    case CANYON: {
                        h = canyon_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 1, 0.5f) * 4.0f;
                        break;
                    }
                    case HILL: {
                        h = hill_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 2, 0.6f) * 4.0f;
                        break;
                    }
                    case PLAIN: {
                        h = plain_func(fx, fy);
                        break;
                    }
                    case MIX: {
                        float h_nn = 0.0f;
                        if (nn) {
                            std::vector<double> inp = { fx, fy, std::sin(fx), std::cos(fy) };
                            auto out = nn->predict(inp);
                            h_nn = float(out[0] * 40.0f - 20.0f);
                        }
                        float h_gauss = float(std::exp(-((fx - 2.0) * (fx - 2.0) + (fy - 2.0) * (fy - 2.0)) / 2.0) * 40.0f - 20.0f);
                        float h_sine = float((std::sin(fx * 2.0) + std::cos(fy * 2.0)) * 10.0f);
                        float h_rand = dist(gen);
                        float h_perlin = fractal_perlin(fx, fy, perlin_perm, 6, 0.5f) * 25.0f;
                        float h_fractal = 0.0f, amp = 10.0f;
                        for (int o = 1; o <= 6; ++o) {
                            h_fractal += std::sin(fx * o + fy * o) * amp;
                            amp *= 0.6f;
                        }
                        h_fractal += fractal_perlin(fx, fy, perlin_perm, 4, 0.5f) * 8.0f;
                        float h_gradient = gradient(fx, fy, 6.0f);
                        float h_mountain = mountain_func(fx, fy);
                        float h_valley = valley_func(fx, fy);
                        float h_plateau = plateau_func(fx, fy);
                        float h_canyon = canyon_func(fx, fy);
                        float h_hill = hill_func(fx, fy);
                        float h_plain = plain_func(fx, fy);

                      // Blend all with weights – większa różnorodność
                        h = 0.07f*h_nn + 0.07f*h_gauss + 0.07f*h_sine + 0.06f*h_rand
                          + 0.09f*h_perlin + 0.09f*h_fractal + 0.09f*h_gradient
                          + 0.11f*h_mountain + 0.09f*h_valley + 0.08f*h_plateau
                          + 0.08f*h_canyon + 0.06f*h_hill + 0.04f*h_plain;

                        break;
                    }
                }
                heights[x][y] = h;
                vertices[x][y].x = float(x);
                vertices[x][y].y = h;
                vertices[x][y].z = float(y);
            }
      
       // Normale
        for (int x = 1; x < width - 1; ++x)
            for (int y = 1; y < height - 1; ++y) {
                float hL = heights[x - 1][y];
                float hR = heights[x + 1][y];
                float hD = heights[x][y - 1];
                float hU = heights[x][y + 1];
                Vertex& v = vertices[x][y];
                v.nx = hL - hR;
                v.ny = 2.0f;
                v.nz = hD - hU;
                float len = sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
                v.nx /= len; v.ny /= len; v.nz /= len;
            }
            
    }
};

// --- Generowanie danych do uczenia NN z różnorodnością ---
void generate_training_data_to_file(const std::string& filename, int count = 5000) {
    std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, TERRAIN_SCALE);
    std::ofstream ofs(filename, std::ios::out);
    if (!ofs.good()) return;

    for (int i = 0; i < count; ++i) {
        double x = dist(gen) / 2;
        double y = dist(gen) * 2;
        double sx = std::sin(x);
        double sy = std::cos(y);

        int type = i % 7; // Różne typy terenu
        double h = 0.0;
        if (type == 0) { // mountain
            h = mountain_func(x, y);
        } else if (type == 1) { // valley
            h = valley_func(x, y);
        } else if (type == 2) { // plateau
            h = plateau_func(x, y);
        } else if (type == 3) { // canyon
            h = canyon_func(x, y);
        } else if (type == 4) { // hill
            h = hill_func(x, y);
        } else if (type == 5) { // plain
            h = plain_func(x, y);
        } else { // mix
            h = 0.25*mountain_func(x,y) + 0.15*valley_func(x,y) + 0.15*plateau_func(x,y)
              + 0.15*canyon_func(x,y) + 0.15*hill_func(x,y) + 0.15*plain_func(x,y);
        }
        ofs << x << " " << y << " " << sx << " " << sy << " " << h << "\n";
    }
    ofs.close();
}

void load_training_data_from_file(const std::string& filename,
                                 std::vector<std::vector<double>>& inputs,
                                 std::vector<std::vector<double>>& targets) {
    std::ifstream ifs(filename);
    if (!ifs.good()) return;
    double x, y, sx, sy, h;
    while (ifs >> x >> y >> sx >> sy >> h) {
        inputs.push_back({ x, y, sx, sy });
        targets.push_back({ h });
    }
    ifs.close();
}

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

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void drawTerrainMesh(const TerrainMesh& mesh) {
    glPushMatrix();
    glTranslatef(-mesh.width / 2.0f, 0.0f, -mesh.height / 2.0f);
    glColor3f(0.7f, 0.7f, 0.7f);
    for (int x = 0; x < mesh.width - 1; ++x) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int y = 0; y < mesh.height; ++y) {
            const Vertex& v1 = mesh.vertices[x][y];
            glNormal3f(v1.nx, v1.ny, v1.nz);
            glVertex3f(v1.x, v1.y, v1.z);
            const Vertex& v2 = mesh.vertices[x + 1][y];
            glNormal3f(v2.nx, v2.ny, v2.nz);
            glVertex3f(v2.x, v2.y, v2.z);
        }
        glEnd();
    }
    glPopMatrix();
}

// --- WinAPI main loop ---
NeuralNetwork nn({ 4, 24, 16, 1 });
TerrainMesh terrain(TERRAIN_WIDTH, TERRAIN_HEIGHT, TERRAIN_SCALE);

bool keys[256] = { 0 };
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
    switch (msg) {
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
        if (wParam == '6') { terrain.algorithm = TerrainMesh::FRACTAL; needsRegenerate = true; }
        if (wParam == '7') { terrain.algorithm = TerrainMesh::GRADIENT; needsRegenerate = true; }
        if (wParam == 'M') { terrain.algorithm = TerrainMesh::MIX; needsRegenerate = true; }
        if (wParam == '8') { terrain.algorithm = TerrainMesh::MOUNTAIN; needsRegenerate = true; }
        if (wParam == '9') { terrain.algorithm = TerrainMesh::VALLEY; needsRegenerate = true; }
        if (wParam == '0') { terrain.algorithm = TerrainMesh::PLATEAU; needsRegenerate = true; }
        if (wParam == 'C') { terrain.algorithm = TerrainMesh::CANYON; needsRegenerate = true; }
        if (wParam == 'H') { terrain.algorithm = TerrainMesh::HILL; needsRegenerate = true; }
        if (wParam == 'P') { terrain.algorithm = TerrainMesh::PLAIN; needsRegenerate = true; }
        if (wParam == 'R') { needsRegenerate = true; }
        if (wParam == 'L') { nn.load("terrain_nn.dat"); terrain.algorithm = TerrainMesh::NN; needsRegenerate = true; }
        if (wParam == 'T') { nn.save("terrain_nn.dat"); }
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

// --- Main ---
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    // 1. Generowanie/zapis danych treningowych do pliku
    std::ifstream training_file("terrain_training_data.dat");
    if (!training_file.good()) {
        MessageBoxA(NULL, "Generowanie danych treningowych do pliku...", "Dane treningowe", MB_OK);
        generate_training_data_to_file("terrain_training_data.dat", 6000);
    }
    training_file.close();

    // 2. Trening sieci NN na tych danych (z ograniczeniem czasu)
    std::ifstream fin("terrain_nn.dat");
    if (fin.good()) {
        nn.load("terrain_nn.dat");
        fin.close();
    } else {
        std::vector<std::vector<double>> train_inputs, train_targets;
        load_training_data_from_file("terrain_training_data.dat", train_inputs, train_targets);
        MessageBoxA(NULL, "Training neural network on realistic terrain (max 1.5 min)...", "Training", MB_OK);
        nn.train(train_inputs, train_targets, 0.07, 120000, 90.0); // 1.5 min limit
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
        while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) goto endloop;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        ProcessKeys();
        if (needsRegenerate) {
            terrain.generate(terrain.algorithm == TerrainMesh::NN || terrain.algorithm == TerrainMesh::MIX ? &nn : nullptr);
            needsRegenerate = false;
        }
        // OpenGL render
        glClearColor(0.21f, 0.32f, 0.51f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        float camX = cameraDist * std::cos(cameraAngle * 3.1415f / 180.0f);
        float camZ = cameraDist * std::sin(cameraAngle * 3.1415f / 180.0f);
        gluLookAt(camX, cameraY, camZ, 0, 0, 0, 0, 1, 0);
        glRotatef(cameraRotX, 1, 0, 0);
        glRotatef(cameraRotY, 0, 1, 0);

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

// --- Koniec pliku. Teraz teren ma więcej typów i realizmu: góry(8), doliny(9), płaskowyże(0), kaniony(C), pagórki(H), równiny(P), mix(M). Klawisze: 1-NN, 2-Gauss, 3-Sine, 4-Random, 5-Perlin, 6-Fractal, 7-Gradient, 8-Góry, 9-Doliny, 0-Płaskowyż, C-Kanion, H-Pagórki, P-Równina, M-MIX, R-regeneruj, T-zapisz NN, L-wczytaj NN
