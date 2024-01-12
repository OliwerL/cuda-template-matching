#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

using namespace std;


__global__ void transparencyPixels(uint8_t* image, int* output, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels; // Zakładamy, że obraz jest w formacie RGB
		if (image[idx] < 10 && image[idx + 1] < 10 && image[idx + 2] < 10) {
			output[y * width + x] = 1;
		}
		else {
			output[y * width + x] = 0;
		}
	}
}

__global__ void grayscaleKernel(uint8_t* d_image, uint8_t* dev_three, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;

		if (channels >= 3) {
			uint8_t R = d_image[idx];
			uint8_t G = d_image[idx + 1];
			uint8_t B = d_image[idx + 2];

			uint8_t gray = R * 0.2126 + G * 0.7152 + B * 0.0722;
			dev_three[idx] = gray;
			dev_three[idx + 1] = gray;
			dev_three[idx + 2] = gray;
		}
	}
}

__global__ void channelExtensionKernel(uint8_t* dev_one, uint8_t* dev_three, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;
		for (int c = 0; c < 3; c++) { //rgb
			dev_three[idx + c] = dev_one[y * width + x];
		}
	}
}

__global__ void channelCompressionKernel(uint8_t* dev_one, uint8_t* dev_three, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * channels;
		dev_one[y * width + x] = dev_three[idx];
	}
}

__global__ void subtractImagesKernel(uint8_t* image, uint8_t* imageA, int* sample_check, int* transparency_array,
	int width, int height, int widthA, int heightA, int treshold) {
	int startX = blockIdx.x * blockDim.x + threadIdx.x;
	int startY = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d, %d\n", startX, startY);
	if (startX <= (width - widthA) && startY <= (height - heightA)) {

		int check_buffer = 0;
		for (int y = 0; y < heightA; y++) {
			for (int x = 0; x < widthA; x++) {
				int largeIdx = ((startY + y) * width + (startX + x)) * 3;
				int smallIdx = (y * widthA + x) * 3;
				int diff = 0;
				//printf("%d, %d\n", image[largeIdx], imageA[smallIdx]);
				if (transparency_array[y * widthA + x] == 1)
				{
					for (int c = 0; c < 3; c++) { //rgb
						diff += abs(image[largeIdx + c] - imageA[smallIdx + c]);
					}
					if (diff < 30)
					{
						check_buffer++;
					}
				}
				else if (transparency_array[y * widthA + x] == 0)
				{
					for (int c = 0; c < 3; c++) { //rgb
						diff += abs(image[largeIdx + c]);
					}
					if (diff > 10)
					{
						check_buffer++;
					}
				}
			}
		}
		if (check_buffer >= treshold)
		{
			sample_check[startY * (width - widthA) + startX] = 1;
			printf("\n Template found; precision: %d / %d, coordinates: %d, %d\n", check_buffer, widthA * heightA, startX, startY);
		}
		else
			sample_check[startY * (width - widthA) + startX] = 0;
	}
}


int main() {
	int widthA, heightA, channelsA, width, height, channels;
	uint8_t* host_imageA = stbi_load("litera.jpg", &widthA, &heightA, &channelsA, 0);
	if (host_imageA == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}

	uint8_t* host_image = stbi_load("testnyga.jpg", &width, &height, &channels, 0);
	if (host_image == nullptr) {
		cerr << "Error loading image to check." << endl;
		return -1;
	}

	// wczytanie wymiarow 
	size_t img_sizeA = widthA * heightA * channelsA;
	uint8_t* dev_imageA;
	size_t img_size = width * height * channels;
	uint8_t* dev_image;

	// ustawienia blokow na gpu
	dim3 blockSize(16, 16);
	dim3 gridSizeA((widthA + blockSize.x - 1) / blockSize.x, (heightA + blockSize.y - 1) / blockSize.y);
	dim3 gridSize((width - widthA + blockSize.x - 1) / blockSize.x, (height - heightA + blockSize.y - 1) / blockSize.y);

	// zrobienie tablicy skladajacej sie z 0 i 1 (tam gdzie 1 znaczy ze jest nasza litera A)
	int* dev_transparency_arr;
	cudaMalloc((void**)&dev_imageA, img_sizeA * sizeof(uint8_t));
	cudaMalloc((void**)&dev_transparency_arr, widthA * heightA * sizeof(int));
	cudaMemcpy(dev_imageA, host_imageA, img_sizeA * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA, dev_transparency_arr, widthA, heightA, channelsA);
	cudaDeviceSynchronize();
	int* transparency_arr = new int[widthA * heightA];
	cudaMemcpy(transparency_arr, dev_transparency_arr, widthA * heightA * sizeof(int), cudaMemcpyDeviceToHost);

	// przerzucenie templatu na greyscale do tablicy host_three_dimension
	uint8_t* dev_three;
	cudaMalloc((void**)&dev_three, img_sizeA * sizeof(uint8_t));
	grayscaleKernel << <gridSizeA, blockSize >> > (dev_imageA, dev_three, widthA, heightA, channelsA);
	cudaDeviceSynchronize();
	uint8_t* host_three_dimension = new uint8_t[img_sizeA];
	cudaMemcpy(host_three_dimension, dev_three, img_sizeA * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	//stbi_write_jpg("test.jpg", widthA, heightA, channelsA, host_three_dimension, 100);

	// przerzucenie host_three_dimension do jednego kanalu
	uint8_t* dev_one;
	cudaMalloc((void**)&dev_one, widthA * heightA * sizeof(uint8_t));
	channelCompressionKernel << <gridSizeA, blockSize >> > (dev_one, dev_three, widthA, heightA, channelsA);
	cudaDeviceSynchronize();
	uint8_t* host_one_dimension = new uint8_t[widthA * heightA];
	cudaMemcpy(host_one_dimension, dev_one, widthA * heightA * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	//uint8_t* dev_rotated_image;
	//cudaMalloc((void**)&dev_rotated_image, img_sizeA);
	//rotate90 << <gridSizeA, blockSize >> > (dev_imageA, dev_rotated_image, width, height, channels);
	//cudaDeviceSynchronize();
	//uint8_t* host_rotated_image = new uint8_t[img_sizeA];
	//cudaMemcpy(host_rotated_image, dev_rotated_image, img_sizeA, cudaMemcpyDeviceToHost);
	//stbi_write_jpg("rotated_image.jpg", heightA, widthA, channelsA, host_rotated_image, 100);
	//cudaFree(dev_rotated_image);

	//sprawdzenie i matching templatu ze zdjeciem
	int* dev_sample_check;
	cudaMalloc((void**)&dev_image, width * height * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&dev_sample_check, (width - widthA) * (height - heightA) * sizeof(int));
	cudaMemcpy(dev_image, host_image, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	int treshold = int(0.98 * widthA * heightA);  // to ma sie zmieniac w zaleznosci od ilosci pixeli naszego sampla, cyli jezeli litera bedzie 100x 100 pixeli to np chcemy sprawdzic czy zgadza sie 90% czyli 0,9*100*100 pixeli jesli tak to wykryte
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA, dev_sample_check, dev_transparency_arr, width, height, widthA, heightA, treshold);
	cudaDeviceSynchronize();
	int* sample_check = new int[(width - widthA) * (height - heightA)];
	cudaMemcpy(sample_check, dev_sample_check, (width - widthA) * (height - heightA) * sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < (width - widthA) * (height - heightA); ++i) {
	//	if (i % (width - widthA) == 0)
	//	{
	//		cout << endl;
	//	}
	//	cout << sample_check[i];
	//}

	// Zwolnienie pamięci GPU
	cudaFree(dev_transparency_arr);
	cudaFree(dev_imageA);
	cudaFree(dev_image);
	cudaFree(dev_sample_check);
	cudaFree(dev_one);
	cudaFree(dev_three);

	// Zwolnienie pamięci hosta
	stbi_image_free(host_image);
	stbi_image_free(host_imageA);
	delete[] transparency_arr;
	delete[] sample_check;
	delete[] host_three_dimension;
	delete[] host_one_dimension;
	//delete[] host_rotated_image;

	cout << endl;
	cout << endl;
	cout << widthA << ' ' << heightA << endl << width << ' ' << height << endl;
	cout << endl;

	return 0;
}
