#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Advanced mathematical constants and macros
#define PI 3.14159265359
#define TAU (2.0 * PI)
#define DEG_TO_RAD(deg) ((deg) * PI / 180.0)
#define RAD_TO_DEG(rad) ((rad) * 180.0 / PI)
#define EPSILON 1e-6
#define MAX_VERTICES 10000
#define MAX_TRIANGLES 5000
#define SCREEN_WIDTH 120
#define SCREEN_HEIGHT 40
#define Z_BUFFER_SIZE (SCREEN_WIDTH * SCREEN_HEIGHT)

// Advanced GPU simulation structures
typedef struct {
    float x, y, z, w;
} Vector4;

typedef struct {
    float x, y, z;
} Vector3;

typedef struct {
    float x, y;
} Vector2;

typedef struct {
    float m[4][4];
} Matrix4x4;

typedef struct {
    Vector4 position;
    Vector3 normal;
    Vector2 texCoord;
    float color;
} Vertex;

typedef struct {
    int v1, v2, v3;
    Vector3 normal;
    float brightness;
} Triangle;

typedef struct {
    char pixel;
    float depth;
    float intensity;
} FrameBuffer;

// GPU State and Pipeline
typedef struct {
    Vertex vertices[MAX_VERTICES];
    Triangle triangles[MAX_TRIANGLES];
    int vertex_count;
    int triangle_count;
    
    Matrix4x4 model_matrix;
    Matrix4x4 view_matrix;
    Matrix4x4 projection_matrix;
    Matrix4x4 mvp_matrix;
    
    FrameBuffer framebuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
    float z_buffer[Z_BUFFER_SIZE];
    
    Vector3 camera_pos;
    Vector3 light_pos;
    Vector3 light_color;
    
    float time;
    int frame_count;
} GPUSimulator;

// Advanced mathematical functions for GPU simulation
Vector4 vec4_create(float x, float y, float z, float w) {
    Vector4 v = {x, y, z, w};
    return v;
}

Vector3 vec3_create(float x, float y, float z) {
    Vector3 v = {x, y, z};
    return v;
}

Vector3 vec3_add(Vector3 a, Vector3 b) {
    return vec3_create(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vector3 vec3_sub(Vector3 a, Vector3 b) {
    return vec3_create(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector3 vec3_scale(Vector3 v, float s) {
    return vec3_create(v.x * s, v.y * s, v.z * s);
}

float vec3_dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 vec3_cross(Vector3 a, Vector3 b) {
    return vec3_create(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float vec3_length(Vector3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3 vec3_normalize(Vector3 v) {
    float len = vec3_length(v);
    if (len < EPSILON) return vec3_create(0, 0, 0);
    return vec3_scale(v, 1.0f / len);
}

// Advanced matrix operations for 3D transformations
Matrix4x4 matrix4x4_identity() {
    Matrix4x4 m = {0};
    m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0f;
    return m;
}

Matrix4x4 matrix4x4_multiply(Matrix4x4 a, Matrix4x4 b) {
    Matrix4x4 result = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return result;
}

Vector4 matrix4x4_multiply_vector(Matrix4x4 m, Vector4 v) {
    return vec4_create(
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w,
        m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w
    );
}

// Advanced 3D transformation matrices
Matrix4x4 create_rotation_x(float angle) {
    Matrix4x4 m = matrix4x4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    
    m.m[1][1] = c;  m.m[1][2] = -s;
    m.m[2][1] = s;  m.m[2][2] = c;
    
    return m;
}

Matrix4x4 create_rotation_y(float angle) {
    Matrix4x4 m = matrix4x4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    
    m.m[0][0] = c;   m.m[0][2] = s;
    m.m[2][0] = -s;  m.m[2][2] = c;
    
    return m;
}

Matrix4x4 create_rotation_z(float angle) {
    Matrix4x4 m = matrix4x4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    
    m.m[0][0] = c;  m.m[0][1] = -s;
    m.m[1][0] = s;  m.m[1][1] = c;
    
    return m;
}

Matrix4x4 create_translation(float x, float y, float z) {
    Matrix4x4 m = matrix4x4_identity();
    m.m[0][3] = x;
    m.m[1][3] = y;
    m.m[2][3] = z;
    return m;
}

Matrix4x4 create_scale(float x, float y, float z) {
    Matrix4x4 m = matrix4x4_identity();
    m.m[0][0] = x;
    m.m[1][1] = y;
    m.m[2][2] = z;
    return m;
}

// Advanced perspective projection with field of view
Matrix4x4 create_perspective(float fov, float aspect, float near, float far) {
    Matrix4x4 m = {0};
    float f = 1.0f / tanf(fov * 0.5f);
    
    m.m[0][0] = f / aspect;
    m.m[1][1] = f;
    m.m[2][2] = (far + near) / (near - far);
    m.m[2][3] = (2.0f * far * near) / (near - far);
    m.m[3][2] = -1.0f;
    
    return m;
}

// Advanced look-at camera matrix
Matrix4x4 create_look_at(Vector3 eye, Vector3 center, Vector3 up) {
    Vector3 f = vec3_normalize(vec3_sub(center, eye));
    Vector3 s = vec3_normalize(vec3_cross(f, up));
    Vector3 u = vec3_cross(s, f);
    
    Matrix4x4 m = matrix4x4_identity();
    
    m.m[0][0] = s.x;
    m.m[1][0] = s.y;
    m.m[2][0] = s.z;
    m.m[0][1] = u.x;
    m.m[1][1] = u.y;
    m.m[2][1] = u.z;
    m.m[0][2] = -f.x;
    m.m[1][2] = -f.y;
    m.m[2][2] = -f.z;
    m.m[0][3] = -vec3_dot(s, eye);
    m.m[1][3] = -vec3_dot(u, eye);
    m.m[2][3] = vec3_dot(f, eye);
    
    return m;
}

// Advanced lighting calculations using Phong reflection model
float calculate_phong_lighting(Vector3 pos, Vector3 normal, Vector3 light_pos, 
                              Vector3 view_pos, Vector3 light_color) {
    Vector3 light_dir = vec3_normalize(vec3_sub(light_pos, pos));
    Vector3 view_dir = vec3_normalize(vec3_sub(view_pos, pos));
    Vector3 reflect_dir = vec3_sub(vec3_scale(normal, 2.0f * vec3_dot(normal, light_dir)), light_dir);
    
    // Ambient component
    float ambient = 0.1f;
    
    // Diffuse component (Lambertian)
    float diffuse = fmaxf(vec3_dot(normal, light_dir), 0.0f);
    
    // Specular component (Phong)
    float spec = powf(fmaxf(vec3_dot(view_dir, reflect_dir), 0.0f), 32.0f);
    
    return ambient + 0.7f * diffuse + 0.3f * spec;
}

// Advanced procedural noise functions for texture generation
float noise_1d(float x) {
    int i = (int)floorf(x);
    float f = x - i;
    
    // Use hash function for pseudorandom values
    float a = sinf(i * 12.9898f) * 43758.5453f;
    float b = sinf((i + 1) * 12.9898f) * 43758.5453f;
    a = a - floorf(a);
    b = b - floorf(b);
    
    // Smooth interpolation
    float u = f * f * (3.0f - 2.0f * f);
    return a * (1.0f - u) + b * u;
}

float noise_3d(Vector3 p) {
    return (noise_1d(p.x) + noise_1d(p.y) + noise_1d(p.z)) / 3.0f;
}

// Fractal Brownian Motion for complex textures
float fbm(Vector3 p, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise_3d(vec3_scale(p, frequency));
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    
    return value;
}

// GPU Simulator initialization
GPUSimulator* gpu_create() {
    GPUSimulator* gpu = (GPUSimulator*)malloc(sizeof(GPUSimulator));
    if (!gpu) return NULL;
    
    memset(gpu, 0, sizeof(GPUSimulator));
    
    gpu->model_matrix = matrix4x4_identity();
    gpu->view_matrix = matrix4x4_identity();
    gpu->projection_matrix = matrix4x4_identity();
    
    gpu->camera_pos = vec3_create(0, 0, 5);
    gpu->light_pos = vec3_create(2, 2, 2);
    gpu->light_color = vec3_create(1, 1, 1);
    
    return gpu;
}

// Clear framebuffer and z-buffer
void gpu_clear_buffers(GPUSimulator* gpu) {
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            gpu->framebuffer[y][x].pixel = ' ';
            gpu->framebuffer[y][x].depth = INFINITY;
            gpu->framebuffer[y][x].intensity = 0.0f;
        }
    }
    
    for (int i = 0; i < Z_BUFFER_SIZE; i++) {
        gpu->z_buffer[i] = INFINITY;
    }
}

// Advanced procedural geometry generation
void gpu_generate_sphere(GPUSimulator* gpu, float radius, int rings, int sectors) {
    gpu->vertex_count = 0;
    gpu->triangle_count = 0;
    
    // Generate sphere vertices using spherical coordinates
    for (int r = 0; r <= rings; r++) {
        float phi = PI * r / rings;  // Latitude angle
        
        for (int s = 0; s <= sectors; s++) {
            float theta = TAU * s / sectors;  // Longitude angle
            
            // Spherical to Cartesian conversion
            float x = radius * sinf(phi) * cosf(theta);
            float y = radius * cosf(phi);
            float z = radius * sinf(phi) * sinf(theta);
            
            if (gpu->vertex_count < MAX_VERTICES) {
                gpu->vertices[gpu->vertex_count].position = vec4_create(x, y, z, 1.0f);
                gpu->vertices[gpu->vertex_count].normal = vec3_normalize(vec3_create(x, y, z));
                gpu->vertices[gpu->vertex_count].texCoord.x = (float)s / sectors;
                gpu->vertices[gpu->vertex_count].texCoord.y = (float)r / rings;
                
                // Apply procedural texture using noise
                Vector3 noise_pos = vec3_scale(vec3_create(x, y, z), 2.0f);
                gpu->vertices[gpu->vertex_count].color = fbm(noise_pos, 4);
                
                gpu->vertex_count++;
            }
        }
    }
    
    // Generate sphere triangles
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            int current = r * (sectors + 1) + s;
            int next = current + sectors + 1;
            
            if (gpu->triangle_count < MAX_TRIANGLES - 1) {
                // First triangle
                gpu->triangles[gpu->triangle_count].v1 = current;
                gpu->triangles[gpu->triangle_count].v2 = next;
                gpu->triangles[gpu->triangle_count].v3 = current + 1;
                gpu->triangle_count++;
                
                // Second triangle
                gpu->triangles[gpu->triangle_count].v1 = current + 1;
                gpu->triangles[gpu->triangle_count].v2 = next;
                gpu->triangles[gpu->triangle_count].v3 = next + 1;
                gpu->triangle_count++;
            }
        }
    }
}

// Generate complex procedural terrain
void gpu_generate_terrain(GPUSimulator* gpu, int width, int height, float scale) {
    gpu->vertex_count = 0;
    gpu->triangle_count = 0;
    
    // Generate heightfield using fractal noise
    for (int z = 0; z < height; z++) {
        for (int x = 0; x < width; x++) {
            float world_x = (x - width * 0.5f) * scale;
            float world_z = (z - height * 0.5f) * scale;
            
            // Generate height using FBM noise
            Vector3 noise_pos = vec3_create(world_x * 0.1f, 0, world_z * 0.1f);
            float height_value = fbm(noise_pos, 6) * 2.0f;
            
            if (gpu->vertex_count < MAX_VERTICES) {
                gpu->vertices[gpu->vertex_count].position = vec4_create(world_x, height_value, world_z, 1.0f);
                
                // Calculate normal using finite differences
                Vector3 pos_x = vec3_create(world_x + 0.1f, 0, world_z);
                Vector3 pos_z = vec3_create(world_x, 0, world_z + 0.1f);
                float height_x = fbm(vec3_scale(pos_x, 0.1f), 6) * 2.0f;
                float height_z = fbm(vec3_scale(pos_z, 0.1f), 6) * 2.0f;
                
                Vector3 tangent_x = vec3_create(0.1f, height_x - height_value, 0);
                Vector3 tangent_z = vec3_create(0, height_z - height_value, 0.1f);
                gpu->vertices[gpu->vertex_count].normal = vec3_normalize(vec3_cross(tangent_x, tangent_z));
                
                gpu->vertices[gpu->vertex_count].color = height_value * 0.5f + 0.5f;
                gpu->vertex_count++;
            }
        }
    }
    
    // Generate terrain triangles
    for (int z = 0; z < height - 1; z++) {
        for (int x = 0; x < width - 1; x++) {
            int i = z * width + x;
            
            if (gpu->triangle_count < MAX_TRIANGLES - 1) {
                // First triangle
                gpu->triangles[gpu->triangle_count].v1 = i;
                gpu->triangles[gpu->triangle_count].v2 = i + width;
                gpu->triangles[gpu->triangle_count].v3 = i + 1;
                gpu->triangle_count++;
                
                // Second triangle
                gpu->triangles[gpu->triangle_count].v1 = i + 1;
                gpu->triangles[gpu->triangle_count].v2 = i + width;
                gpu->triangles[gpu->triangle_count].v3 = i + width + 1;
                gpu->triangle_count++;
            }
        }
    }
}

// Advanced vertex shader simulation
Vector4 vertex_shader(GPUSimulator* gpu, Vertex* vertex) {
    // Apply model transformations with time-based animation
    Matrix4x4 rotation_y = create_rotation_y(gpu->time * 0.5f);
    Matrix4x4 rotation_x = create_rotation_x(sinf(gpu->time) * 0.2f);
    Matrix4x4 scale = create_scale(1.0f + sinf(gpu->time * 2.0f) * 0.1f, 
                                  1.0f + cosf(gpu->time * 1.5f) * 0.1f, 
                                  1.0f);
    
    gpu->model_matrix = matrix4x4_multiply(matrix4x4_multiply(rotation_y, rotation_x), scale);
    
    // Create view matrix (orbiting camera)
    float cam_angle = gpu->time * 0.3f;
    Vector3 camera_pos = vec3_create(cosf(cam_angle) * 8.0f, 4.0f, sinf(cam_angle) * 8.0f);
    gpu->view_matrix = create_look_at(camera_pos, vec3_create(0, 0, 0), vec3_create(0, 1, 0));
    gpu->camera_pos = camera_pos;
    
    // Create projection matrix
    gpu->projection_matrix = create_perspective(DEG_TO_RAD(60), 
                                               (float)SCREEN_WIDTH / SCREEN_HEIGHT, 
                                               0.1f, 100.0f);
    
    // Combine transformations
    gpu->mvp_matrix = matrix4x4_multiply(matrix4x4_multiply(gpu->projection_matrix, gpu->view_matrix), 
                                        gpu->model_matrix);
    
    // Transform vertex position
    return matrix4x4_multiply_vector(gpu->mvp_matrix, vertex->position);
}

// Advanced fragment shader with multiple lighting models
char fragment_shader(GPUSimulator* gpu, Vector3 world_pos, Vector3 normal, float color, float depth) {
    // Calculate lighting using advanced Phong model
    float lighting = calculate_phong_lighting(world_pos, normal, gpu->light_pos, 
                                            gpu->camera_pos, gpu->light_color);
    
    // Add atmospheric perspective
    float distance = vec3_length(vec3_sub(world_pos, gpu->camera_pos));
    float fog_factor = expf(-distance * 0.1f);
    lighting *= fog_factor;
    
    // Combine procedural texture with lighting
    float final_intensity = (color * 0.7f + 0.3f) * lighting;
    
    // Map intensity to ASCII characters for visualization
    final_intensity = fmaxf(0.0f, fminf(1.0f, final_intensity));
    
    const char intensity_map[] = " .,-~:;=!*#$@";
    int char_index = (int)(final_intensity * (sizeof(intensity_map) - 2));
    return intensity_map[char_index];
}

// Advanced rasterization with sub-pixel accuracy
void rasterize_triangle(GPUSimulator* gpu, int t_idx) {
    Triangle* tri = &gpu->triangles[t_idx];
    
    // Get transformed vertices
    Vector4 v1_clip = vertex_shader(gpu, &gpu->vertices[tri->v1]);
    Vector4 v2_clip = vertex_shader(gpu, &gpu->vertices[tri->v2]);  
    Vector4 v3_clip = vertex_shader(gpu, &gpu->vertices[tri->v3]);
    
    // Perspective divide
    if (v1_clip.w <= 0 || v2_clip.w <= 0 || v3_clip.w <= 0) return;
    
    Vector3 v1_ndc = vec3_create(v1_clip.x/v1_clip.w, v1_clip.y/v1_clip.w, v1_clip.z/v1_clip.w);
    Vector3 v2_ndc = vec3_create(v2_clip.x/v2_clip.w, v2_clip.y/v2_clip.w, v2_clip.z/v2_clip.w);
    Vector3 v3_ndc = vec3_create(v3_clip.x/v3_clip.w, v3_clip.y/v3_clip.w, v3_clip.z/v3_clip.w);
    
    // Screen space conversion
    int x1 = (int)((v1_ndc.x + 1.0f) * 0.5f * SCREEN_WIDTH);
    int y1 = (int)((1.0f - v1_ndc.y) * 0.5f * SCREEN_HEIGHT);
    int x2 = (int)((v2_ndc.x + 1.0f) * 0.5f * SCREEN_WIDTH);
    int y2 = (int)((1.0f - v2_ndc.y) * 0.5f * SCREEN_HEIGHT);
    int x3 = (int)((v3_ndc.x + 1.0f) * 0.5f * SCREEN_WIDTH);
    int y3 = (int)((1.0f - v3_ndc.y) * 0.5f * SCREEN_HEIGHT);
    
    // Bounding box
    int min_x = fmax(0, fmin(fmin(x1, x2), x3));
    int max_x = fmin(SCREEN_WIDTH - 1, fmax(fmax(x1, x2), x3));
    int min_y = fmax(0, fmin(fmin(y1, y2), y3));
    int max_y = fmin(SCREEN_HEIGHT - 1, fmax(fmax(y1, y2), y3));
    
    // Rasterize using barycentric coordinates
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            // Calculate barycentric coordinates
            float denom = (float)((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            if (fabsf(denom) < EPSILON) continue;
            
            float w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
            float w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
            float w3 = 1.0f - w1 - w2;
            
            // Check if point is inside triangle
            if (w1 >= 0 && w2 >= 0 && w3 >= 0) {
                // Interpolate depth
                float z = w1 * v1_ndc.z + w2 * v2_ndc.z + w3 * v3_ndc.z;
                
                // Z-buffer test
                if (z < gpu->framebuffer[y][x].depth) {
                    gpu->framebuffer[y][x].depth = z;
                    
                    // Interpolate vertex attributes
                    Vector3 world_pos = vec3_add(vec3_add(
                        vec3_scale(vec3_create(gpu->vertices[tri->v1].position.x,
                                             gpu->vertices[tri->v1].position.y,
                                             gpu->vertices[tri->v1].position.z), w1),
                        vec3_scale(vec3_create(gpu->vertices[tri->v2].position.x,
                                             gpu->vertices[tri->v2].position.y,
                                             gpu->vertices[tri->v2].position.z), w2)),
                        vec3_scale(vec3_create(gpu->vertices[tri->v3].position.x,
                                             gpu->vertices[tri->v3].position.y,
                                             gpu->vertices[tri->v3].position.z), w3));
                    
                    Vector3 normal = vec3_normalize(vec3_add(vec3_add(
                        vec3_scale(gpu->vertices[tri->v1].normal, w1),
                        vec3_scale(gpu->vertices[tri->v2].normal, w2)),
                        vec3_scale(gpu->vertices[tri->v3].normal, w3)));
                    
                    float color = w1 * gpu->vertices[tri->v1].color + 
                                 w2 * gpu->vertices[tri->v2].color + 
                                 w3 * gpu->vertices[tri->v3].color;
                    
                    // Run fragment shader
                    gpu->framebuffer[y][x].pixel = fragment_shader(gpu, world_pos, normal, color, z);
                }
            }
        }
    }
}

// Main rendering pipeline
void gpu_render_frame(GPUSimulator* gpu) {
    gpu_clear_buffers(gpu);
    
    // Update time for animations
    gpu->time += 0.016f;  // ~60 FPS
    gpu->frame_count++;
    
    // Update light position for dynamic lighting
    gpu->light_pos = vec3_create(cosf(gpu->time) * 3.0f, 
                                2.0f + sinf(gpu->time * 2.0f), 
                                sinf(gpu->time) * 3.0f);
    
    // Render all triangles
    for (int i = 0; i < gpu->triangle_count; i++) {
        rasterize_triangle(gpu, i);
    }
}

// Display framebuffer to console
void gpu_present(GPUSimulator* gpu) {
    // Clear screen
    printf("\033[2J\033[H");
    
    // Display performance info
    printf("GPU Simulator - Frame: %d, Time: %.2fs, Vertices: %d, Triangles: %d\n", 
           gpu->frame_count, gpu->time, gpu->vertex_count, gpu->triangle_count);
    printf("Camera: (%.1f, %.1f, %.1f), Light: (%.1f, %.1f, %.1f)\n\n",
           gpu->camera_pos.x, gpu->camera_pos.y, gpu->camera_pos.z,
           gpu->light_pos.x, gpu->light_pos.y, gpu->light_pos.z);
    
    // Render framebuffer
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            printf("%c", gpu->framebuffer[y][x].pixel);
        }
        printf("\n");
    }
}

// Cleanup
void gpu_destroy(GPUSimulator* gpu) {
    if (gpu) {
        free(gpu);
    }
}

// Demo scenes
void demo_spinning_sphere(GPUSimulator* gpu) {
    gpu_generate_sphere(gpu, 2.0f, 20, 20);
}

void demo_procedural_terrain(GPUSimulator* gpu) {
    gpu_generate_terrain(gpu, 20, 20, 0.5f);
}

// Main simulation loop
int main() {
    GPUSimulator* gpu = gpu_create();
    if (!gpu) {
        printf("Failed to create GPU simulator\n");
        return 1;
    }
    
    printf("Advanced GPU Simulator with Mathematical 3D Engine\n");
    printf("Choose demo:\n");
    printf("1. Spinning Sphere with Procedural Texture\);