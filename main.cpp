#include <windows.h>
#include <GL/gl.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <random>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

// Stałe i globalne zmienne
#define TERRAIN_WIDTH 256
#define TERRAIN_HEIGHT 256


std::random_device rd;  // uzyskujemy zródło entropii
std::mt19937 gen(rd());  // inicjujemy generator

float normals[TERRAIN_WIDTH][TERRAIN_HEIGHT][3];


float cameraX=50, cameraY=50, cameraZ=150;
float cameraYaw=-45, cameraPitch=-20;
const float cameraYawMin=-360, cameraYawMax=360;
const float cameraPitchMin=-89, cameraPitchMax=89;

int width=800, height=600; // rozmiar okna
float frequency=0.1f;
unsigned int seed=0;
float *heightMap=NULL;

// Perlin noise variables
float *perlinGradient=NULL;
unsigned char *perlinPermutation=NULL;

// Fractal parameters
int fractalDepth=4; // głębokość fraktala
float fractalAmplitude=10.0f; // amplituda fraktala

// Keys
bool keys[256]={false};

// Funkcje
void GenerateTerrain();
void CalculateNormals();
void DrawTerrain();
void SetupLighting();
void SetViewMatrix();
void HandleInput();
float GetHeightAt(int x,int z);
void FreeTerrain();
void DrawAxes();
void SetProjection(int width, int height);
void SetCustomProjection(float fovY, float aspect, float zNear, float zFar);
void GeneratePerlinGradient();
void InitializePermutation();
float PerlinNoise(float x, float z);
float Fade(float t);
float Lerp(float a, float b, float t);
float Gradient(int hash, float x, float z);
float FractalNoise(float x, float z, int depth, float amplitude);
float PolynomialInterpolate(float t, float h0, float h1, float h2, float h3);
void SmoothTerrain(int iterations);

// Windows functions declarations
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
void EnableOpenGL(HWND hWnd, HDC *hDC, HGLRC *hRC);
void DisableOpenGL(HWND hWnd, HDC hDC, HGLRC hRC);

// Main
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASS wc={0};
    wc.style=CS_OWNDC;
    wc.lpfnWndProc=WndProc;
    wc.hInstance=hInst;
    wc.lpszClassName="TerrainWindow";

    RegisterClass(&wc);
    HWND hWnd=CreateWindow(wc.lpszClassName,"Procedural Terrain 3D",
        WS_OVERLAPPEDWINDOW|WS_VISIBLE,
        CW_USEDEFAULT,CW_USEDEFAULT,800,600,
        NULL,NULL,hInst,NULL);

    HDC hDC; HGLRC hRC;
    EnableOpenGL(hWnd,&hDC,&hRC);

    seed=(unsigned int)time(NULL);
    GeneratePerlinGradient();
    InitializePermutation();
    GenerateTerrain();
    SmoothTerrain(1); // dwie iteracje wygładzania
    CalculateNormals();
    SetupLighting();

    glClearColor(0.1f, 0.2f, 0.4f, 1.0f); 

    MSG msg;
    BOOL running=TRUE;

    while(running)
    {
        while(PeekMessage(&msg,NULL,0,0,PM_REMOVE))
        {
            if(msg.message==WM_QUIT)
                running=FALSE;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        HandleInput();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        SetProjection(width, height);
        glMatrixMode(GL_MODELVIEW);
        SetViewMatrix();

        
        DrawTerrain();

        SwapBuffers(hDC);
        // Sleep(10);
    }

    FreeTerrain();
    DisableOpenGL(hWnd,hDC,hRC);
    DestroyWindow(hWnd);
    return 0;
}



void CalculateNormalAt(int x, int z, float* normal)
{
    float heightL=GetHeightAt(x-1,z);
    float heightR=GetHeightAt(x+1,z);
    float heightD=GetHeightAt(x,z-1);
    float heightU=GetHeightAt(x,z+1);

    normal[0]=heightL - heightR;
    normal[1]=2.0f;
    normal[2]=heightD - heightU;

    float len=sqrtf(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if(len==0) len=1;
    normal[0]/=len;
    normal[1]/=len;
    normal[2]/=len;
}

void CalculateNormals()
{
    for(int z=0;z<TERRAIN_HEIGHT;z++)
    {
        for(int x=0;x<TERRAIN_WIDTH;x++)
        {
            CalculateNormalAt(x,z,normals[x][z]);
        }
    }
}

// Setup lighting
void SetupLighting()
{
    GLfloat ambientLight[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat diffuseLight[] = {0.7f, 0.7f, 0.7f, 1.0f};
    GLfloat lightPos[] = {50.0f, 100.0f, 50.0f, 1.0f};

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glEnable(GL_LIGHT0);
}

// Set projection
void SetProjection(int width, int height)
{
    if(height==0) height=1;
    float aspect = (float)width / (float)height;
    float fovY = 45.0f;
    float zNear=1.0f;
    float zFar=2000.0f;
    SetCustomProjection(fovY, aspect, zNear, zFar);
}

void SetCustomProjection(float fovY, float aspect, float zNear, float zFar)
{
    float f = 1.0f / tanf(fovY * M_PI/360.0f);
    float m[16]={
        f/aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zFar+zNear)/(zNear - zFar), (2*zFar*zNear)/(zNear - zFar),
        0, 0, -1, 0
    };
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glLoadMatrixf(m);
    glMatrixMode(GL_MODELVIEW);
}

// WndProc
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_SIZE:
            width=LOWORD(lParam);
            height=HIWORD(lParam);
            return 0;
        case WM_KEYDOWN:
            keys[wParam]=true;
            if(wParam==VK_ESCAPE) PostQuitMessage(0);
            return 0;
        case WM_KEYUP:
            keys[wParam]=false;
            return 0;
        default:
            return DefWindowProc(hWnd,msg,wParam,lParam);
    }
}

// OpenGL setup
void EnableOpenGL(HWND hWnd, HDC *hDC, HGLRC *hRC)
{
    PIXELFORMATDESCRIPTOR pfd={0};
    pfd.nSize=sizeof(pfd);
    pfd.nVersion=1;
    pfd.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType=PFD_TYPE_RGBA;
    pfd.cColorBits=24;
    pfd.cDepthBits=24;
    pfd.iLayerType=PFD_MAIN_PLANE;

    *hDC=GetDC(hWnd);
    int format=ChoosePixelFormat(*hDC,&pfd);
    SetPixelFormat(*hDC,format,&pfd);
    *hRC=wglCreateContext(*hDC);
    wglMakeCurrent(*hDC,*hRC);

    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.1f, 0.2f, 0.4f, 1.0f);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
}

void DisableOpenGL(HWND hWnd, HDC hDC, HGLRC hRC)
{
    wglMakeCurrent(NULL,NULL);
    wglDeleteContext(hRC);
    ReleaseDC(hWnd,hDC);
}

// Generate Perlin Gradient
void GeneratePerlinGradient() {
    int size=256;
    perlinGradient=(float*)malloc(sizeof(float)*size*2);
    for(int i=0;i<size;i++)
    {
        float angle=(float)(rand()%360)*M_PI/180.0f;
        perlinGradient[i*2]=cosf(angle);
        perlinGradient[i*2+1]=sinf(angle);
    }
}

// Initialize permutation array
void InitializePermutation() {
    int size=256;
    perlinPermutation=(unsigned char*)malloc(sizeof(unsigned char)*size);
    for(int i=0;i<size;i++) perlinPermutation[i]=i;

    for(int i=size-1;i>0;i--)
    {
        int j=rand()%(i+1);
        unsigned char temp=perlinPermutation[i];
        perlinPermutation[i]=perlinPermutation[j];
        perlinPermutation[j]=temp;
    }

    unsigned char* permDup=(unsigned char*)malloc(sizeof(unsigned char)*512);
    for(int i=0;i<512;i++) permDup[i]=perlinPermutation[i%size];

    free(perlinPermutation);
    perlinPermutation=permDup;
}

// Fade function
float Fade(float t)
{
    return t*t*t*(t*(t*6-15)+10);
}

// Linear interpolation
float Lerp(float a, float b, float t)
{
    return a + t*(b - a);
}

// Gradient calculation
float Gradient(int hash, float x, float z)
{
    int h=hash & 255;
    float gx=perlinGradient[h*2];
    float gz=perlinGradient[h*2+1];
    return gx*x + gz*z;
}

// Perlin noise function
float PerlinNoise(float x, float z)
{
    int xi=(int)floorf(x)&255;
    int zi=(int)floorf(z)&255;
    float xf=x - floorf(x);
    float zf=z - floorf(z);

    float u=Fade(xf);
    float v=Fade(zf);

    int aa=perlinPermutation[xi]+zi;
    int ab=perlinPermutation[xi]+zi+1;
    int ba=perlinPermutation[xi+1]+zi;
    int bb=perlinPermutation[xi+1]+zi+1;

    float gradAA=Gradient(perlinPermutation[aa],xf,zf);
    float gradBA=Gradient(perlinPermutation[ba],xf-1,zf);
    float gradAB=Gradient(perlinPermutation[ab],xf,zf-1);
    float gradBB=Gradient(perlinPermutation[bb],xf-1,zf-1);

    float lerpX1=Lerp(gradAA,gradBA,u);
    float lerpX2=Lerp(gradAB,gradBB,u);

    return Lerp(lerpX1,lerpX2,v);
}

// Fractal noise
float FractalNoise(float x, float z, int depth, float amplitude)
{
    float total=0;
    float freq=1.0f;
    float amp=amplitude;
    for(int i=0;i<depth;i++)
    {
        total+=PerlinNoise(x*freq,z*freq)*amp;
        freq*=2.0f;
        amp*=0.5f;
    }
    return total;
}

void GenerateTerrain()
{
    if(heightMap) free(heightMap);
    heightMap = (float*)malloc(sizeof(float) * TERRAIN_WIDTH * TERRAIN_HEIGHT);
    srand(seed);

    float lakeCenterX = TERRAIN_WIDTH / 2.0f;
    float lakeCenterZ = TERRAIN_HEIGHT / 2.0f;
    float lakeRadius = 50.0f;
    float oceanLevel = -10.0f; // poziom oceanu

    for (int z = 0; z < TERRAIN_HEIGHT; z++)
    {
        for (int x = 0; x < TERRAIN_WIDTH; x++)
        {
            float nx = x * frequency;
            float nz = z * frequency;

            float h = 0;

            // Fraktalny szum z większą głębokością i bardziej naturalnym rozkładem
            for (int o = 0; o < fractalDepth; o++)
            {
                float freqval = powf(2.0f, o);
                float amp = fractalAmplitude * powf(0.5f, o) * 0.5f; // zmniejszona amplituda
                h += PerlinNoise(nx * freqval, nz * freqval) * amp;
            }

            // Dodanie bardziej naturalnych wzgórz i pagórków
            float hill1 = powf(sinf(nx * M_PI * 0.3f), 3) * 12 + powf(cosf(nz * M_PI * 0.2f), 3) * 12;
            h += hill1;

            float mountain = powf(sinf(nx * M_PI * 0.1f), 4) * 20;
            h += mountain;

            // Dodanie drobnych nierówności i szumu, aby tereny nie były zbyt gładkie
            float smallDetails = ((rand() % 10) - 5) * 0.2f; // zakres od -1 do +1
            h += smallDetails;

            // Wygładzanie i eliminacja małych wypustek
            if (h > 15) h=15; // limit wysokości, aby nie było zbyt stromych szczytów
            if (h < -20) h=-20; // ograniczenie głębokości

            // Jezioro w centrum, z delikatnym spadkiem
            float distToLake = sqrtf((x - lakeCenterX)*(x - lakeCenterX) + (z - lakeCenterZ)*(z - lakeCenterZ));
            if (distToLake < lakeRadius)
            {
                float depression = -sinf(distToLake * 0.2f) * 20;
                h = fminf(h, depression);
            }

            // Ustawienie poziomu oceanu
            if (h < oceanLevel)
            {
                h = oceanLevel + ((float)(rand() % 3) - 1);
            }

            // Brzegi są bardziej płaskie, aby nie było wielkości wzgórza w postaci szpilki
            if (x < 30 || x > TERRAIN_WIDTH - 30 || z < 30 || z > TERRAIN_HEIGHT - 30)
            {
                h *= 0.5f;
            }

            // Drobne szczegóły, aby teren był bardziej naturalny
            h += ((rand() % 10) - 5) * 0.2f;

            heightMap[z * TERRAIN_WIDTH + x] = h;
        }
    }
}


void SmoothTerrain(int iterations)
{
	std::uniform_real_distribution<double> dist_real(0.0, 5.0);
    double los = dist_real(gen);
    
    float* temp = (float*)malloc(sizeof(float)*TERRAIN_WIDTH*TERRAIN_HEIGHT);
    for (int iter=0; iter<iterations; iter++)
    {
        for (int z=0; z<TERRAIN_HEIGHT; z++)
        {
            for (int x=0; x<TERRAIN_WIDTH; x++)
            {
                float sum=0;
                int count=0;
                for (int dz=-1; dz<=1; dz++)
                {
                    for (int dx=-1; dx<=1; dx++)
                    {
                        int nx=x+dx;
                        int nz=z+dz;
                        if (nx>=0 && nx<TERRAIN_WIDTH && nz>=0 && nz<TERRAIN_HEIGHT)
                        {
                            sum+=heightMap[nz*TERRAIN_WIDTH + nx];
                            count++;
                        }
                    }
                }
                float average = sum/count;
                // Dodaj delikatny losowy szum, aby teren był bardziej poszarpany
                double noise = ((los - 0.5) + ( 0.1f * los)) * 2; // zakres od -5 do +5
                temp[z*TERRAIN_WIDTH + x]=average + noise;
            }
        }
        memcpy(heightMap, temp, sizeof(float)*TERRAIN_WIDTH*TERRAIN_HEIGHT);
    }
    free(temp);
}


// Get height at (x,z) with interpolation
float GetHeightAt(int x,int z)
{
    if(!heightMap) return 0;
    if(x<0) x=0;
    if(x>=TERRAIN_WIDTH) x=TERRAIN_WIDTH-1;
    if(z<0) z=0;
    if(z>=TERRAIN_HEIGHT) z=TERRAIN_HEIGHT-1;
    return heightMap[z*TERRAIN_WIDTH+x];
}

// Optional: Calculate normal for lighting
void CalculateNormal(int x, int z, float* normal)
{
    float heightL=GetHeightAt(x-1,z);
    float heightR=GetHeightAt(x+1,z);
    float heightD=GetHeightAt(x,z-1);
    float heightU=GetHeightAt(x,z+1);

    normal[0]=heightL - heightR;
    normal[1]=2.0f;
    normal[2]=heightD - heightU;

    float len=sqrtf(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if(len==0) len=1;
    normal[0]/=len;
    normal[1]/=len;
    normal[2]/=len;
}

void DrawTerrain()
{
    float scaleX=1.0f;
    float scaleZ=1.0f;
    float heightScale=2.0f;

    glColor3f(0.3f,0.9f,0.3f);
    for(int z=0;z<TERRAIN_HEIGHT-1;z++)
    {
        glBegin(GL_TRIANGLE_STRIP);
        for(int x=0;x<TERRAIN_WIDTH;x++)
        {
            float h1=GetHeightAt(x,z)*heightScale;
            float h2=GetHeightAt(x,z+1)*heightScale;

            // Normalne dla dolnego wierzchołka
            float* n1=normals[x][z];
            glNormal3f(n1[0], n1[1], n1[2]);
            glVertex3f(x*scaleX, h1, z*scaleZ);

            // Normalne dla górnego wierzchołka
            float* n2=normals[x][z+1];
            glNormal3f(n2[0], n2[1], n2[2]);
            glVertex3f(x*scaleX, h2, (z+1)*scaleZ);
        }
        glEnd();
    }
}

// Set view matrix
void SetViewMatrix()
{
    // Clamp angles
    if(cameraYaw > cameraYawMax) cameraYaw=cameraYawMax;
    if(cameraYaw < cameraYawMin) cameraYaw=cameraYawMin;
    if(cameraPitch > cameraPitchMax) cameraPitch=cameraPitchMax;
    if(cameraPitch < cameraPitchMin) cameraPitch=cameraPitchMin;

    float radYaw=cameraYaw*M_PI/180.0f;
    float radPitch=cameraPitch*M_PI/180.0f;

    float dirX=cosf(radPitch)*sinf(radYaw);
    float dirY=sinf(radPitch);
    float dirZ=cosf(radPitch)*cosf(radYaw);

    float lookX=cameraX+dirX;
    float lookY=cameraY+dirY;
    float lookZ=cameraZ+dirZ;

    float upX=0, upY=1, upZ=0;

    // lookAt matrix
    float fX=lookX - cameraX;
    float fY=lookY - cameraY;
    float fZ=lookZ - cameraZ;
    float fLen=sqrtf(fX*fX+fY*fY+fZ*fZ);
    if(fLen==0) fLen=1;
    fX/=fLen; fY/=fLen; fZ/=fLen;

    float sX,sY,sZ;
    sX=upY*fZ - upZ*fY;
    sY=upZ*fX - upX*fZ;
    sZ=upX*fY - upY*fX;
    float sLen=sqrtf(sX*sX+sY*sY+sZ*sZ);
    if(sLen==0) sLen=1;
    sX/=sLen; sY/=sLen; sZ/=sLen;

    float uX=fY*sZ - fZ*sY;
    float uY=fZ*sX - fX*sZ;
    float uZ=fX*sY - fY*sX;

    float m[16]={
        sX, uX, -fX, 0,
        sY, uY, -fY, 0,
        sZ, uZ, -fZ, 0,
        0,   0,    0,  1
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixf(m);
    glTranslatef(-cameraX, -cameraY, -cameraZ);
}


// Handle input
void HandleInput()
{
    float moveSpeed=1.0f;
    float rotSpeed=1.0f;

    if(GetAsyncKeyState(VK_LEFT)&0x8000) cameraYaw -= rotSpeed;
    if(GetAsyncKeyState(VK_RIGHT)&0x8000) cameraYaw += rotSpeed;
    if(GetAsyncKeyState(VK_UP)&0x8000) cameraPitch += rotSpeed;
    if(GetAsyncKeyState(VK_DOWN)&0x8000) cameraPitch -= rotSpeed;

    if(cameraYaw > cameraYawMax) cameraYaw=cameraYawMax;
    if(cameraYaw < cameraYawMin) cameraYaw=cameraYawMin;
    if(cameraPitch > cameraPitchMax) cameraPitch=cameraPitchMax;
    if(cameraPitch < cameraPitchMin) cameraPitch=cameraPitchMin;

    float radYaw=cameraYaw*M_PI/180.0f;

    if(GetAsyncKeyState('W')&0x8000)
    {
        cameraX+=sinf(radYaw)*moveSpeed;
        cameraZ-=cosf(radYaw)*moveSpeed;
    }
    if(GetAsyncKeyState('S')&0x8000)
    {
        cameraX-=sinf(radYaw)*moveSpeed;
        cameraZ+=cosf(radYaw)*moveSpeed;
    }
    if(GetAsyncKeyState('A')&0x8000)
    {
        float leftYaw=radYaw - M_PI/2;
        cameraX+=sinf(leftYaw)*moveSpeed;
        cameraZ-=cosf(leftYaw)*moveSpeed;
    }
    if(GetAsyncKeyState('D')&0x8000)
    {
        float rightYaw=radYaw + M_PI/2;
        cameraX+=sinf(rightYaw)*moveSpeed;
        cameraZ-=cosf(rightYaw)*moveSpeed;
    }

    if(GetAsyncKeyState(VK_SPACE)&0x8000) cameraY+=moveSpeed;
    if(GetAsyncKeyState(VK_SHIFT)&0x8000) cameraY-=moveSpeed;
}

// Free terrain
void FreeTerrain()
{
    if(heightMap) free(heightMap);
    heightMap=NULL;
}

// Note: You can add more functions like DrawAxes() if needed.
