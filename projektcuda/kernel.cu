#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

__global__ void processImageKernel(uint8_t* d_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        if (channels >= 3) {
            uint8_t R = d_image[idx];
            uint8_t G = d_image[idx + 1];
            uint8_t B = d_image[idx + 2];

            uint8_t gray = R * 0.2126 + G * 0.7152 + B * 0.0722;

            // Ustawienie wartości RGB na wartość skali szarości
            d_image[idx] = gray;       // R
            d_image[idx + 1] = gray;   // G
            d_image[idx + 2] = gray;   // B
        }
       

    }
}

__global__ void gaussianBlurKernel(uint8_t* d_image, uint8_t* d_result, int width, int height, int channels) {
    const float gaussianKernel[5][5] = {
    {1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f},
    {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
    {7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f},
    {4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f},
    {1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f}
    };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x >= width || y >= height) return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    int kernelRadius = 2; // Radius jądra 5x5 jest równy 2

    for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
        for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int idx = (py * width + px) * channels;

            float kernelValue = gaussianKernel[ky + kernelRadius][kx + kernelRadius];
            sumR += d_image[idx] * kernelValue;
            sumG += d_image[idx + 1] * kernelValue;
            sumB += d_image[idx + 2] * kernelValue;
        }
    }

    
    d_result[idx] = min(max(static_cast<int>(sumR), 0), 255);
    d_result[idx + 1] = min(max(static_cast<int>(sumG), 0), 255);
    d_result[idx + 2] = min(max(static_cast<int>(sumB), 0), 255);

   
}


int main() {
    int width, height, channels;
    uint8_t* h_image = stbi_load("fortnite.png", &width, &height, &channels, 0);
    if (h_image == nullptr) {
        cerr << "Error loading image." << endl;
        return -1;
    }

    size_t img_size = width * height * channels;
    uint8_t* d_image, * d_result;

    // Alokacja pamięci dla oryginalnego i tymczasowego obrazu
    cudaMalloc(&d_image, img_size);
    
    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);

    // Ustawienie rozmiaru siatki i bloku
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Uruchomienie kernela do konwersji na skalę szarości
    processImageKernel << <gridSize, blockSize >> > (d_image, width, height, channels);
    cudaDeviceSynchronize();
    

    // Kopiowanie przetworzonych danych z powrotem do pamięci hosta
    cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost);
    stbi_write_png("grayscale.png", width, height, channels, h_image, width * channels);

    cudaMalloc(&d_result, img_size);
    cudaMemcpy(d_result, h_image, img_size, cudaMemcpyHostToDevice);
    gaussianBlurKernel << <gridSize, blockSize >> > (d_image, d_result, width, height, channels);
    cudaDeviceSynchronize();

    // (Opcjonalnie) Kopiowanie oryginalnego obrazu w skali szarości z powrotem do pamięci hosta
    cudaMemcpy(h_image, d_result, img_size, cudaMemcpyDeviceToHost);

    // (Opcjonalnie) Zapisanie oryginalnego obrazu w skali szarości
    stbi_write_png("bllured.png", width, height, channels, h_image, width * channels);

    // Zwolnienie pamięci GPU
    cudaFree(d_image);
    cudaFree(d_result);

    // Zwolnienie pamięci hosta
    stbi_image_free(h_image);

    return 0;
}

