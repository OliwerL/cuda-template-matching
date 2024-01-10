#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

__global__ void readRGBValuesKernel(uint8_t* d_image, int width, int height, int channels, float3* d_rgbValues) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        if (channels >= 3) {
            float R = d_image[idx];
            float G = d_image[idx + 1];
            float B = d_image[idx + 2];
            d_rgbValues[y * width + x] = make_float3(R, G, B);
        }
    }
}

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
    {1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f}               //test komenatrz
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
            int idxx = (py * width + px) * channels;

            float kernelValue = gaussianKernel[ky + kernelRadius][kx + kernelRadius];
            sumR += d_image[idxx] * kernelValue;
            sumG += d_image[idxx + 1] * kernelValue;
            sumB += d_image[idxx + 2] * kernelValue;
        }
    }
    d_result[idx] = min(max(static_cast<int>(sumR), 0), 255);
    d_result[idx + 1] = min(max(static_cast<int>(sumG), 0), 255);
    d_result[idx + 2] = min(max(static_cast<int>(sumB), 0), 255);
}

__global__ void sepiaKernel(uint8_t* d_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        if (channels >= 3) {
            float R = d_image[idx];
            float G = d_image[idx + 1];
            float B = d_image[idx + 2];

            float sepiaR = min(255.0f, (R * 0.393f) + (G * 0.769f) + (B * 0.189f));
            float sepiaG = min(255.0f, (R * 0.349f) + (G * 0.686f) + (B * 0.168f));
            float sepiaB = min(255.0f, (R * 0.272f) + (G * 0.534f) + (B * 0.131f));


            d_image[idx] = static_cast<uint8_t>(sepiaR);
            d_image[idx + 1] = static_cast<uint8_t>(sepiaG);
            d_image[idx + 2] = static_cast<uint8_t>(sepiaB);
        }
    }
}

__global__ void negativeKernel(uint8_t* d_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        if (channels >= 3) {
            d_image[idx] = 255 - d_image[idx];       // R
            d_image[idx + 1] = 255 - d_image[idx + 1];   // G
            d_image[idx + 2] = 255 - d_image[idx + 2];   // B
        }
    }
}


int main() {
    int widthA, heightA, channelsA;
    uint8_t* h_imageA = stbi_load("litera.jpg", &widthA, &heightA, &channelsA, 0);
    if (h_imageA == nullptr) {
        cerr << "Error loading image 'litera.png'." << endl;
        return -1;
    }

    size_t img_sizeA = widthA * heightA * channelsA;
    uint8_t* d_imageA;
    float3* d_rgbValuesA;
    cudaMalloc(&d_imageA, img_sizeA);
    cudaMalloc(&d_rgbValuesA, widthA * heightA * sizeof(float3));
    cudaMemcpy(d_imageA, h_imageA, img_sizeA, cudaMemcpyHostToDevice);

    // Ustaw rozmiar siatki i bloku dla obrazu 'litera.png'
    dim3 blockSizeA(16, 16);
    dim3 gridSizeA((widthA + blockSizeA.x - 1) / blockSizeA.x, (heightA + blockSizeA.y - 1) / blockSizeA.y);

    // Uruchom kernel do wczytywania wartości RGB
    readRGBValuesKernel << <gridSizeA, blockSizeA >> > (d_imageA, widthA, heightA, channelsA, d_rgbValuesA);
    cudaDeviceSynchronize();

    // Skopiuj wyniki z powrotem do pamięci hosta
    float3* h_rgbValuesA = new float3[widthA * heightA];                //wartosci rgb litery A
    cudaMemcpy(h_rgbValuesA, d_rgbValuesA, widthA * heightA * sizeof(float3), cudaMemcpyDeviceToHost);

    for (int i = 0; i < widthA * heightA; ++i) {
        cout << "R: " << h_rgbValuesA[i].x << " G: " << h_rgbValuesA[i].y << " B: " << h_rgbValuesA[i].z << endl;
    }

    cudaFree(d_imageA);
    cudaFree(d_rgbValuesA);
    delete[] h_rgbValuesA;
    stbi_image_free(h_imageA);


    int width, height, channels;
    uint8_t* h_image = stbi_load("aaa1.jpg", &width, &height, &channels, 0);
    if (h_image == nullptr) {
        cerr << "Error loading image." << endl;
        return -1;
    }

    size_t img_size = width * height * channels;
    uint8_t* d_image, * d_result;

    // Alokacja pamięci dla oryginalnego obrazu
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

    // Alokacja pamięci dla rozmytego obrazu
    cudaMalloc(&d_result, img_size);

    // Uruchomienie kernela do rozmycia gaussowskiego
    gaussianBlurKernel << <gridSize, blockSize >> > (d_image, d_result, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(h_image, d_result, img_size, cudaMemcpyDeviceToHost);
    stbi_write_png("blurred.png", width, height, channels, h_image, width * channels);

    // Uruchomienie kernela do filtru sepia
    sepiaKernel << <gridSize, blockSize >> > (d_image, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost);
    stbi_write_png("sepia.png", width, height, channels, h_image, width * channels);

    negativeKernel << <gridSize, blockSize >> > (d_image, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost);
    stbi_write_png("negative.png", width, height, channels, h_image, width * channels);


    // Zwolnienie pamięci GPU
    cudaFree(d_image);
    cudaFree(d_result);

    // Zwolnienie pamięci hosta
    stbi_image_free(h_image);

    return 0;
}
