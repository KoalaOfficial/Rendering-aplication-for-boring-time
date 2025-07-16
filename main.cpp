
#include <windows.h>
#include <gl/gl.h>
#include "terrain.h"
#include "renderer.h"

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Globalne zmienne
TerrainGenerator* terrain = nullptr;
TerrainRenderer* renderer = nullptr;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Rejestracja klasy okna
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "TerrainWindow";
    
    if (!RegisterClass(&wc)) return 0;
    
    // Tworzenie okna
    HWND hWnd = CreateWindow(
        wc.lpszClassName,
        "Procedural Terrain 3D",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        800, 600,
        NULL, NULL,
        hInstance, NULL
    );
    
    if (!hWnd) return 0;
    
    // Inicjalizacja
    terrain = new TerrainGenerator();
    renderer = new TerrainRenderer(terrain);
    renderer->Initialize(hWnd);
    
    // Rozpoczęcie generowania terenu z ekranem ładowania
    renderer->EnableLoadingScreen(true);
    
    // Generowanie terenu w osobnym wątku
    CreateThread(NULL, 0, [](LPVOID param) -> DWORD {
        TerrainGenerator* t = (TerrainGenerator*)param;
        
        // Generowanie terenu
        t->GenerateTerrain();
        t->SmoothTerrain(2);
        t->CalculateNormals();
        
        // Zapis do pliku
        t->SaveToFile("terrain.bin");
        
        renderer->EnableLoadingScreen(false);
        return 0;
    }, terrain, 0, NULL);
    
    // Główna pętla
    MSG msg;
    BOOL running = TRUE;
    HDC hDC = GetDC(hWnd);
    
    while (running) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = FALSE;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        
        renderer->HandleInput();
        renderer->Render(hDC);
    }
    
    // Czyszczenie
    delete renderer;
    delete terrain;
    
    return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
            
        case WM_SIZE:
            if (renderer) {
                renderer->Resize(LOWORD(lParam), HIWORD(lParam));
            }
            return 0;
            
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
            }
            return 0;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }
}
