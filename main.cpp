// main.cpp
#include <algorithm> 
#include <windows.h>
#include <gl/gl.h>
#include <gl/glu.h>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <ctime>
#include <chrono>
#include <sstream>

#define CLAMP(x, upper, lower) (std::min(upper, std::max(x, lower)))

#define TERRAIN_WIDTH 256
#define TERRAIN_HEIGHT 256
#define TERRAIN_SCALE 5.0f

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

LPCSTR WINDOW_CLASS = "TerrainNN3D";
LPCSTR WINDOW_TITLE = "3D Terrain Mesh & Neural Net";


void HandleTerrainAlgorithmSelection(WPARAM wParam);
void CaptureMouse(HWND hwnd);
void ReleaseMouse();

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
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : 0);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}
float perlin2d(float x, float y, const std::vector<int>& p) {
    int xi = int(floor(x)) & 255;
    int yi = int(floor(y)) & 255;

    float xf = x - floor(x);
    float yf = y - floor(y);

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
float fractal_perlin(float x, float y, const std::vector<int>& p, int octaves = 6, float persistence = 0.55f, float lacunarity = 2.0f) {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maxValue = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        total += perlin2d(x * frequency, y * frequency, p) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return total / maxValue;
}

// --- Realistyczne generatory terenów i biomów ---
float mountain_func(float x, float y) {
    float r = std::sqrt((x-1.5)*(x-1.5)+(y-1.5)*(y-1.5));
    float peak = std::exp(-(r*r)*0.7)*45.0f + std::sin(x*6 + y*4)*11.0f;
    float ridge = std::abs(std::sin(x*10 + y*9)*13.0f);
    return peak + ridge - 12.0f;
}
float valley_func(float x, float y) {
    float r = std::sqrt((x-2.8)*(x-2.8)+(y-0.5)*(y-0.5));
    return -std::exp(-r * 1.2) * 28.0f + std::sin(x*2.0 + y*2.0)*3.5f;
}
float plateau_func(float x, float y) {
    float base = std::exp(-((x-2.2)*(x-2.2)+(y-2.2)*(y-2.2))*1.1)*30.0f;
    float flat = std::max(0.0f, float(1.0 - std::abs(x-2.2)*0.7 - std::abs(y-2.2)*0.7))*8.0f;
    return base + flat + std::sin(x*1.7 + y*1.7)*4.0f;
}
float canyon_func(float x, float y) {
    float c = std::sin(x*8.5)*std::cos(y*4.0)*12.0f - std::abs(x-1.7)*22.0f;
    float noise = std::sin(x*2.5+y*2.8)*3.0f;
    return c + noise;
}
float hill_func(float x, float y) {
    return std::sin(x*7.0 + y*5.0)*17.0f + std::sin(x*7.5)*7.0f + std::cos(y*7.0)*7.0f;
}
float plain_func(float x, float y) {
    return std::sin(x*1.3)*2.8f + std::sin(y*1.6)*2.8f;
}
float desert_func(float x, float y, const std::vector<int>& p) {
    float sand = fractal_perlin(x, y, p, 3, 0.55f) * 7.0f + std::sin(x*3.0 + y*2.5)*3.0f;
    float ripple = std::sin(x*9.0 + y*8.0)*1.5f;
    return sand + ripple - 2.0f;
}
float forest_func(float x, float y, const std::vector<int>& p) {
    float base = fractal_perlin(x, y, p, 5, 0.7f) * 9.0f + std::sin(x*2.0 + y*2.0)*2.0f;
    float bumps = std::sin(x*12.0 + y*11.0)*2.5f;
    return base + bumps;
}
float tundra_func(float x, float y, const std::vector<int>& p) {
    float flat = fractal_perlin(x, y, p, 4, 0.6f) * 6.0f;
    float wave = std::sin(x*2.1)*1.0f + std::cos(y*1.7)*1.0f;
    return flat + wave - 1.0f;
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

    enum GenAlgorithm {
        NN, GAUSS, PERLIN, SINE, RANDOM,
        FRACTAL, GRADIENT, MIX,
        MOUNTAIN, VALLEY, PLATEAU, CANYON, HILL, PLAIN,
        DESERT, FOREST, TUNDRA
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
        if (perlin_perm.empty()) initPerlin();

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
                        h = fractal_perlin(fx, fy, perlin_perm, 8, 0.53f) * 30.0f;
                        break;
                    }
                    case FRACTAL: {
                        float sum = 0.0f, amp = 10.0f;
                        for (int o = 1; o <= 8; ++o) {
                            sum += std::sin(fx * o + fy * o) * amp;
                            amp *= 0.53f;
                        }
                        sum += fractal_perlin(fx, fy, perlin_perm, 7, 0.62f) * 12.0f;
                        h = sum;
                        break;
                    }
                    case GRADIENT: {
                        h = (fx + fy) * 8.0f;
                        break;
                    }
                    case MOUNTAIN: {
                        h = mountain_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 4, 0.65f) * 10.0f;
                        break;
                    }
                    case VALLEY: {
                        h = valley_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 3, 0.67f) * 8.0f;
                        break;
                    }
                    case PLATEAU: {
                        h = plateau_func(fx, fy) + std::sin(fx*3.5+fy*3.5)*3.2f;
                        break;
                    }
                    case CANYON: {
                        h = canyon_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 2, 0.65f) * 6.0f;
                        break;
                    }
                    case HILL: {
                        h = hill_func(fx, fy) + fractal_perlin(fx, fy, perlin_perm, 3, 0.67f) * 6.0f;
                        break;
                    }
                    case PLAIN: {
                        h = plain_func(fx, fy);
                        break;
                    }
                    case DESERT: {
                        h = desert_func(fx, fy, perlin_perm);
                        break;
                    }
                    case FOREST: {
                        h = forest_func(fx, fy, perlin_perm);
                        break;
                    }
                    case TUNDRA: {
                        h = tundra_func(fx, fy, perlin_perm);
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
                        float h_perlin = fractal_perlin(fx, fy, perlin_perm, 8, 0.55f) * 32.0f;
                        float h_fractal = 0.0f, amp = 11.0f;
                        for (int o = 1; o <= 8; ++o) {
                            h_fractal += std::sin(fx * o + fy * o) * amp;
                            amp *= 0.54f;
                        }
                        h_fractal += fractal_perlin(fx, fy, perlin_perm, 6, 0.58f) * 9.5f;
                        float h_gradient = (fx + fy) * 8.0f;
                        float h_mountain = mountain_func(fx, fy);
                        float h_valley = valley_func(fx, fy);
                        float h_plateau = plateau_func(fx, fy);
                        float h_canyon = canyon_func(fx, fy);
                        float h_hill = hill_func(fx, fy);
                        float h_plain = plain_func(fx, fy);
                        float h_desert = desert_func(fx, fy, perlin_perm);
                        float h_forest = forest_func(fx, fy, perlin_perm);
                        float h_tundra = tundra_func(fx, fy, perlin_perm);

                        // Blend all types
                        h = 0.06f*h_nn + 0.06f*h_gauss + 0.06f*h_sine + 0.05f*h_rand
                          + 0.07f*h_perlin + 0.07f*h_fractal + 0.07f*h_gradient
                          + 0.11f*h_mountain + 0.08f*h_valley + 0.07f*h_plateau
                          + 0.08f*h_canyon + 0.07f*h_hill + 0.04f*h_plain
                          + 0.05f*h_desert + 0.05f*h_forest + 0.05f*h_tundra;
                        break;
                    }
                }
                heights[x][y] = h;
                vertices[x][y].x = float(x);
                vertices[x][y].y = h;
                vertices[x][y].z = float(y);
            }

        // --- Wygladzanie powierzchni (smooth) ---
        std::vector<std::vector<float>> smoothed(width, std::vector<float>(height, 0.0f));
        for (int x = 1; x < width-1; ++x)
            for (int y = 1; y < height-1; ++y) {
                smoothed[x][y] = (heights[x][y]*0.5f +
                                  heights[x-1][y]*0.1f + heights[x+1][y]*0.1f +
                                  heights[x][y-1]*0.1f + heights[x][y+1]*0.1f +
                                  heights[x-1][y-1]*0.05f + heights[x+1][y+1]*0.05f +
                                  heights[x-1][y+1]*0.05f + heights[x+1][y-1]*0.05f);
            }
        for (int x = 1; x < width-1; ++x)
            for (int y = 1; y < height-1; ++y)
                heights[x][y] = smoothed[x][y];

        // --- Normale
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

// --- Kamera gracza (first-person) ---
struct Camera {
    float x = 128.0f, y = 25.0f, z = 128.0f;
    float pitch = 0.0f, yaw = 0.0f;
    float speed = 1.7f;
    bool onGround = true;

    void update(bool* keys, TerrainMesh& terrain) {
        float forward = 0, right = 0, up = 0;
        if (keys['W']) forward += speed;
        if (keys['S']) forward -= speed;
        if (keys['A']) right -= speed;
        if (keys['D']) right += speed;
        if (keys[VK_SPACE]) up += speed*0.7f;
        if (keys[VK_CONTROL]) up -= speed*0.7f;

        float radYaw = yaw * 3.14159f / 180.0f;
        float radPitch = pitch * 3.14159f / 180.0f;
        float dx = cos(radYaw) * cos(radPitch);
        float dz = sin(radYaw) * cos(radPitch);

        x += dx * forward + dz * right;
        z += dz * forward - dx * right;
        y += up;

        x = std::max(2.0f, std::min(x, float(terrain.width-2)));
        z = std::max(2.0f, std::min(z, float(terrain.height-2)));
        int ix = int(x), iz = int(z);
        if (ix >= 0 && iz >= 0 && ix < terrain.width && iz < terrain.height) {
            float targetY = terrain.heights[ix][iz] + 2.9f;
            y += (targetY - y) * 0.35f;
        }
    }
};

Camera playerCam;
POINT lastMouse = {0,0};
bool mouseCaptured = false;

// --- Generowanie danych do uczenia NN z różnorodnością ---
void generate_training_data_to_file(const std::string& filename, int count = 7000) {
    std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, TERRAIN_SCALE);
    std::ofstream ofs(filename, std::ios::out);
    if (!ofs.good()) return;

    for (int i = 0; i < count; ++i) {
        double x = dist(gen) / 2;
        double y = dist(gen) * 2;
        double sx = std::sin(x);
        double sy = std::cos(y);

        int type = i % 10;
        double h = 0.0;
        if (type == 0) h = mountain_func(x, y);
        else if (type == 1) h = valley_func(x, y);
        else if (type == 2) h = plateau_func(x, y);
        else if (type == 3) h = canyon_func(x, y);
        else if (type == 4) h = hill_func(x, y);
        else if (type == 5) h = plain_func(x, y);
        else if (type == 6) h = desert_func(x, y, std::vector<int>(256,0));
        else if (type == 7) h = forest_func(x, y, std::vector<int>(256,0));
        else if (type == 8) h = tundra_func(x, y, std::vector<int>(256,0));
        else {
            h = 0.17*mountain_func(x,y) + 0.11*valley_func(x,y) + 0.11*plateau_func(x,y)
              + 0.11*canyon_func(x,y) + 0.11*hill_func(x,y) + 0.11*plain_func(x,y)
              + 0.14*desert_func(x,y,std::vector<int>(256,0)) + 0.14*forest_func(x,y,std::vector<int>(256,0)) + 0.14*tundra_func(x,y,std::vector<int>(256,0));
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

// --- OpenGL setup ---
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

// --- Rysowanie terenu z kolorami biomów ---
void drawTerrainMesh(const TerrainMesh& mesh) {
    glPushMatrix();
    glTranslatef(-mesh.width / 2.0f, 0.0f, -mesh.height / 2.0f);
    for (int x = 0; x < mesh.width - 1; ++x) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int y = 0; y < mesh.height; ++y) {
            const Vertex& v1 = mesh.vertices[x][y];
            float h = mesh.heights[x][y];
            if (h > 22.0f) glColor3f(0.62f, 0.54f, 0.47f);
            else if (h > 12.0f) glColor3f(0.48f, 0.65f, 0.34f);
            else if (h > 4.0f) glColor3f(0.84f, 0.74f, 0.46f);
            else if (h < -16.0f) glColor3f(0.77f, 0.58f, 0.31f);
            else if (h < -5.0f) glColor3f(0.78f, 0.78f, 0.85f);
            else glColor3f(0.49f, 0.74f, 0.72f);
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

// --- Sterowanie i obsługa myszki ---
bool keys[256] = { 0 };
bool needsRegenerate = false;
HWND global_hwnd = NULL;



// Funkcja obsługująca klawisze i ruch myszy, aktualizująca kamerę i stan gry
void ProcessKeys() {
    if (mouseCaptured) {
        POINT pt;
        GetCursorPos(&pt);
        float dx = static_cast<float>(pt.x - lastMouse.x);
        float dy = static_cast<float>(pt.y - lastMouse.y);

        playerCam.yaw += dx * 0.18f;
        playerCam.pitch -= dy * 0.16f;
        playerCam.pitch = CLAMP(playerCam.pitch, -80.0f, 80.0f);

        lastMouse = pt;

        RECT rect;
        GetWindowRect(global_hwnd, &rect);
        int cx = (rect.left + rect.right) / 2;
        int cy = (rect.top + rect.bottom) / 2;
        SetCursorPos(cx, cy);
        lastMouse.x = cx;
        lastMouse.y = cy;
    }

    playerCam.update(keys, terrain);
}

// Funkcja obsługi komunikatów Windows
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            break;

        case WM_KEYDOWN:
            keys[wParam] = true;
            HandleTerrainAlgorithmSelection(wParam);
            break;

        case WM_KEYUP:
            keys[wParam] = false;
            break;

        case WM_SIZE:
            setupOpenGL(LOWORD(lParam), HIWORD(lParam));
            break;

        case WM_LBUTTONDOWN:
            CaptureMouse(hwnd);
            break;

        case WM_LBUTTONUP:
            ReleaseMouse();
            break;

        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

void HandleTerrainAlgorithmSelection(WPARAM wParam) {
    switch (wParam) {
        case '1': terrain.algorithm = TerrainMesh::NN; break;
        case '2': terrain.algorithm = TerrainMesh::GAUSS; break;
        case '3': terrain.algorithm = TerrainMesh::SINE; break;
        case '4': terrain.algorithm = TerrainMesh::RANDOM; break;
        case '5': terrain.algorithm = TerrainMesh::PERLIN; break;
        case '6': terrain.algorithm = TerrainMesh::FRACTAL; break;
        case '7': terrain.algorithm = TerrainMesh::GRADIENT; break;
        case 'M': terrain.algorithm = TerrainMesh::MIX; break;
        case '8': terrain.algorithm = TerrainMesh::MOUNTAIN; break;
        case '9': terrain.algorithm = TerrainMesh::VALLEY; break;
        case '0': terrain.algorithm = TerrainMesh::PLATEAU; break;
        case 'C': terrain.algorithm = TerrainMesh::CANYON; break;
        case 'H': terrain.algorithm = TerrainMesh::HILL; break;
        case 'P': terrain.algorithm = TerrainMesh::PLAIN; break;
        case 'D': terrain.algorithm = TerrainMesh::DESERT; break;
        case 'F': terrain.algorithm = TerrainMesh::FOREST; break;
        case 'T': terrain.algorithm = TerrainMesh::TUNDRA; break;
        case 'R': needsRegenerate = true; break;
        case 'L':
            nn.load("terrain_nn.dat");
            terrain.algorithm = TerrainMesh::NN;
            needsRegenerate = true;
            break;
        case 'S':
            nn.save("terrain_nn.dat");
            break;
    }
}

void CaptureMouse(HWND hwnd) {
    mouseCaptured = true;
    SetCapture(hwnd);
    RECT rect;
    GetWindowRect(hwnd, &rect);
    int cx = (rect.left + rect.right) / 2;
    int cy = (rect.top + rect.bottom) / 2;
    SetCursorPos(cx, cy);
    lastMouse.x = cx;
    lastMouse.y = cy;
    ShowCursor(FALSE);
}

void ReleaseMouse() {
    mouseCaptured = false;
    ReleaseCapture();
    ShowCursor(TRUE);
}


// --- Main ---
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    // 1. Generowanie/zapis danych treningowych do pliku
    std::ifstream training_file("terrain_training_data.dat");
    if (!training_file.good()) {
        MessageBoxA(NULL, "Generowanie danych treningowych do pliku...", "Dane treningowe", MB_OK);
        generate_training_data_to_file("terrain_training_data.dat", 7000);
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
        MessageBoxA(NULL, "Training neural network on realistic terrain (max 2 min)...", "Training", MB_OK);
        nn.train(train_inputs, train_targets, 0.07, 150000, 120.0);
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
    global_hwnd = hwnd; // do obsługi myszki

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

        // Kamera gracza (first-person)
        float radYaw = playerCam.yaw * 3.14159f / 180.0f;
        float radPitch = playerCam.pitch * 3.14159f / 180.0f;
        float lookX = playerCam.x + cos(radYaw) * cos(radPitch);
        float lookY = playerCam.y + sin(radPitch);
        float lookZ = playerCam.z + sin(radYaw) * cos(radPitch);
        gluLookAt(playerCam.x, playerCam.y, playerCam.z,
                  lookX, lookY, lookZ,
                  0, 1, 0);

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

