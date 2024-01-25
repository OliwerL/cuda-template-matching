#define STB_IMAGE_IMPLEMENTATION
#define _USE_MATH_DEFINES
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

__global__ void transparencyPixels(uint8_t* image, int* output, int width, int height, int channels);
__global__ void grayscaleKernel(uint8_t* d_image, uint8_t* dev_three, int width, int height, int channels);
__global__ void channelExtensionKernel(uint8_t* dev_one, uint8_t* dev_three, int width, int height, int channels);
__global__ void channelCompressionKernel(uint8_t* dev_one, uint8_t* dev_three, int width, int height, int channels);
__global__ void subtractImagesKernel(uint8_t* image, uint8_t* imageA, int* sample_check, int* transparency_array, int width, int height, int widthA, int heightA, int treshold);
__global__ void rotate90degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height);
__global__ void rotate180degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height);
__global__ void rotate270degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height);
__global__ void rotate45degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void rotate135degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void rotate225degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void rotate315degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void increase_letters(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void decrease_letters(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight);
__global__ void bounding_box(uint8_t* input, uint8_t* output, int width, int height, int widthA, int heightA, int* sample_check, int* top_left, int samples);
__global__ void find_top_left(int width, int height, int widthA, int heightA, int* sample_check, int* top_left, int* global_counter);
__global__ void find_top_left90(int width, int height, int widthA90, int heightA90, int* sample_check, int* top_left, int* global_counter);


int main() {
	int widthA, heightA, channelsA, widthA90, heightA90, channelsA90, widthA45, heightA45, channelsA45, width, height, channels, widthAinc, heightAinc, channelsAinc, widthAdec, heightAdec, channelsAdec;
	uint8_t* host_imageA = stbi_load("litera.jpg", &widthA, &heightA, &channelsA, 0);
	if (host_imageA == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}



	uint8_t* host_image = stbi_load("labedzM.jpg", &width, &height, &channels, 0);
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
	//dim3 gridSizeA90((widthA90 + blockSize.x - 1) / blockSize.x, (heightA90 + blockSize.y - 1) / blockSize.y);
	dim3 gridSize((width - widthA + blockSize.x - 1) / blockSize.x, (height - heightA + blockSize.y - 1) / blockSize.y);
	dim3 gridSizeBB((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


	// zrobienie tablicy skladajacej sie z 0 i 1 (tam gdzie 1 znaczy ze jest nasza litera A)
	int* dev_transparency_arr;
	//cudaMalloc((void**)&dev_imageA, img_sizeA * sizeof(uint8_t));
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



	// obrócenie litery o 90stopni
	uint8_t* dev_rotated90d_image;
	cudaMalloc((void**)&dev_rotated90d_image, widthA * heightA * sizeof(uint8_t));
	rotate90degree << <gridSizeA, blockSize >> > (dev_one, dev_rotated90d_image, widthA, heightA);
	cudaDeviceSynchronize();
	uint8_t* host_rotated90d_image = new uint8_t[widthA * heightA];
	cudaMemcpy(host_rotated90d_image, dev_rotated90d_image, widthA * heightA * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated90.jpg", heightA, widthA, 1, host_rotated90d_image, 100);

	// obrócenie litery o 180stopni
	uint8_t* dev_rotated180d_image;
	cudaMalloc((void**)&dev_rotated180d_image, widthA * heightA * sizeof(uint8_t));
	rotate180degree << <gridSizeA, blockSize >> > (dev_one, dev_rotated180d_image, widthA, heightA);
	cudaDeviceSynchronize();
	uint8_t* host_rotated180d_image = new uint8_t[widthA * heightA];
	cudaMemcpy(host_rotated180d_image, dev_rotated180d_image, widthA * heightA * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated180.jpg", widthA, heightA, 1, host_rotated180d_image, 100);

	// obrócenie litery o 270stopni
	uint8_t* dev_rotated270d_image;
	cudaMalloc((void**)&dev_rotated270d_image, widthA * heightA * sizeof(uint8_t));
	rotate270degree << <gridSizeA, blockSize >> > (dev_one, dev_rotated270d_image, widthA, heightA);
	cudaDeviceSynchronize();
	uint8_t* host_rotated270d_image = new uint8_t[widthA * heightA];
	cudaMemcpy(host_rotated270d_image, dev_rotated270d_image, widthA * heightA * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated270.jpg", heightA, widthA, 1, host_rotated270d_image, 100);

	// obrócenie litery o 45stopni
	uint8_t* dev_rotated45d_image;
	int newDimension = static_cast<int>(sqrt(widthA * widthA + heightA * heightA));
	int height_45d = newDimension + 2;
	int width_45d = newDimension + 1;
	dim3 gridSize45d((width_45d + blockSize.x - 1) / blockSize.x, (height_45d + blockSize.y - 1) / blockSize.y);
	cudaMalloc((void**)&dev_rotated45d_image, width_45d * height_45d * sizeof(uint8_t));
	cudaMemset(dev_rotated45d_image, 255, width_45d * height_45d * sizeof(uint8_t));
	rotate45degree << <gridSize45d, blockSize >> > (dev_one, dev_rotated45d_image, widthA, heightA, width_45d, height_45d);
	cudaDeviceSynchronize();
	uint8_t* host_rotated45d_image = new uint8_t[width_45d * height_45d];
	cudaMemcpy(host_rotated45d_image, dev_rotated45d_image, width_45d * height_45d * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated45.jpg", width_45d, height_45d, 1, host_rotated45d_image, 100);

	// obrócenie litery o 135 stopni
	uint8_t* dev_rotated135d_image;
	int newDimension135 = static_cast<int>(sqrt(widthA * widthA + heightA * heightA));
	int height_135d = newDimension135 + 2;
	int width_135d = newDimension135 + 1;
	dim3 gridSize135d((width_135d + blockSize.x - 1) / blockSize.x, (height_135d + blockSize.y - 1) / blockSize.y);

	cudaMalloc((void**)&dev_rotated135d_image, width_135d * height_135d * sizeof(uint8_t));
	cudaMemset(dev_rotated135d_image, 255, width_135d * height_135d * sizeof(uint8_t));

	rotate135degree << <gridSize135d, blockSize >> > (dev_one, dev_rotated135d_image, widthA, heightA, width_135d, height_135d);
	cudaDeviceSynchronize();

	uint8_t* host_rotated135d_image = new uint8_t[width_135d * height_135d];
	cudaMemcpy(host_rotated135d_image, dev_rotated135d_image, width_135d * height_135d * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated135.jpg", width_135d, height_135d, 1, host_rotated135d_image, 100);

	// obrócenie litery o 225 stopni
	uint8_t* dev_rotated225d_image;
	int newDimension225 = static_cast<int>(sqrt(widthA * widthA + heightA * heightA));
	int height_225d = newDimension225 + 2;
	int width_225d = newDimension225 + 1;
	dim3 gridSize225d((width_225d + blockSize.x - 1) / blockSize.x, (height_225d + blockSize.y - 1) / blockSize.y);

	cudaMalloc((void**)&dev_rotated225d_image, width_225d * height_225d * sizeof(uint8_t));
	cudaMemset(dev_rotated225d_image, 255, width_225d * height_225d * sizeof(uint8_t));

	rotate225degree << <gridSize225d, blockSize >> > (dev_one, dev_rotated225d_image, widthA, heightA, width_225d, height_225d);
	cudaDeviceSynchronize();

	uint8_t* host_rotated225d_image = new uint8_t[width_225d * height_225d];
	cudaMemcpy(host_rotated225d_image, dev_rotated225d_image, width_225d * height_225d * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated225.jpg", width_225d, height_225d, 1, host_rotated225d_image, 100);



	// obrócenie litery o 315stopni
	uint8_t* dev_rotated315d_image;
	//int newDimension = static_cast<int>(sqrt(widthA * widthA + heightA * heightA));
	int height_315d = newDimension + 2;
	int width_315d = newDimension + 1;
	dim3 gridSize315d((width_315d + blockSize.x - 1) / blockSize.x, (height_315d + blockSize.y - 1) / blockSize.y);
	cudaMalloc((void**)&dev_rotated315d_image, width_315d * height_315d * sizeof(uint8_t));
	cudaMemset(dev_rotated315d_image, 255, width_315d * height_315d * sizeof(uint8_t));
	rotate315degree << <gridSize315d, blockSize >> > (dev_one, dev_rotated315d_image, widthA, heightA, width_315d, height_315d);
	cudaDeviceSynchronize();
	uint8_t* host_rotated315d_image = new uint8_t[width_315d * height_315d];
	cudaMemcpy(host_rotated315d_image, dev_rotated315d_image, width_315d * height_315d * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("rotated315.jpg", width_315d, height_315d, 1, host_rotated315d_image, 100);


	// zwiększanie litery
	uint8_t* dev_increase_image;
	int height_increase = heightA * 1.5;
	int width_increase = widthA * 1.5;
	dim3 gridSizeincrease((width_increase + blockSize.x - 1) / blockSize.x, (height_increase + blockSize.y - 1) / blockSize.y);
	cudaMalloc((void**)&dev_increase_image, width_increase * height_increase * sizeof(uint8_t));
	cudaMemset(dev_increase_image, 255, width_increase * height_increase * sizeof(uint8_t));
	increase_letters << <gridSizeincrease, blockSize >> > (dev_one, dev_increase_image, widthA, heightA, width_increase, height_increase);
	cudaDeviceSynchronize();
	uint8_t* host_increase_image = new uint8_t[width_increase * height_increase];
	cudaMemcpy(host_increase_image, dev_increase_image, width_increase * height_increase * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("increase.jpg", width_increase, height_increase, 1, host_increase_image, 100);

	// zmniejszanie litery
	uint8_t* dev_decrease_image;
	int height_decrease = heightA / 1.5;
	int width_decrease = widthA / 1.5;
	dim3 gridSizedecrease((width_decrease + blockSize.x - 1) / blockSize.x, (height_decrease + blockSize.y - 1) / blockSize.y);
	cudaMalloc((void**)&dev_decrease_image, width_decrease * height_decrease * sizeof(uint8_t));
	cudaMemset(dev_decrease_image, 255, width_decrease * height_decrease * sizeof(uint8_t));
	decrease_letters << <gridSizedecrease, blockSize >> > (dev_one, dev_decrease_image, widthA, heightA, width_decrease, height_decrease);
	cudaDeviceSynchronize();
	uint8_t* host_decrease_image = new uint8_t[width_decrease * height_decrease];
	cudaMemcpy(host_decrease_image, dev_decrease_image, width_decrease * height_decrease * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("decrease.jpg", width_decrease, height_decrease, 1, host_decrease_image, 100);

	//wczytanie zrotowanych zdjec

	uint8_t* host_imageA90 = stbi_load("rotated90.jpg", &widthA90, &heightA90, &channelsA90, 0);
	if (host_imageA90 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA270 = stbi_load("rotated270.jpg", &widthA90, &heightA90, &channelsA90, 0);
	if (host_imageA270 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA180 = stbi_load("rotated180.jpg", &widthA, &heightA, &channelsA, 0);
	if (host_imageA180 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA45 = stbi_load("rotated45.jpg", &widthA45, &heightA45, &channelsA45, 0);
	if (host_imageA45 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA135 = stbi_load("rotated135.jpg", &widthA45, &heightA45, &channelsA45, 0);
	if (host_imageA135 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA225 = stbi_load("rotated225.jpg", &widthA45, &heightA45, &channelsA45, 0);
	if (host_imageA225 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}
	uint8_t* host_imageA315 = stbi_load("rotated315.jpg", &widthA45, &heightA45, &channelsA45, 0);
	if (host_imageA315 == nullptr) {
		cerr << "Error loading sample image." << endl;
		return -1;
	}



	size_t img_sizeA90 = widthA90 * heightA90 * channelsA90;
	size_t img_sizeA45 = widthA45 * heightA45 * channelsA45;


	uint8_t* dev_imageA90;
	uint8_t* dev_imageA270;
	uint8_t* dev_imageA180;
	uint8_t* dev_imageA45;
	uint8_t* dev_imageA135;
	uint8_t* dev_imageA225;
	uint8_t* dev_imageA315;
	uint8_t* dev_imageAinc;
	uint8_t* dev_imageAdec;

	//transparency
	int* dev_transparency_arr90;
	cudaMalloc((void**)&dev_imageA90, img_sizeA90 * sizeof(uint8_t));
	cudaMalloc((void**)&dev_transparency_arr90, widthA90 * heightA90 * sizeof(int));
	cudaMemcpy(dev_imageA90, host_imageA90, img_sizeA90 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA90, dev_transparency_arr90, widthA90, heightA90, channelsA90);
	cudaDeviceSynchronize();
	int* transparency_arr90 = new int[widthA90 * heightA90];
	cudaMemcpy(transparency_arr90, dev_transparency_arr90, widthA90 * heightA90 * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr270;
	cudaMalloc((void**)&dev_transparency_arr270, widthA90 * heightA90 * sizeof(int));
	cudaMalloc((void**)&dev_imageA270, img_sizeA90 * sizeof(uint8_t));
	cudaMemcpy(dev_imageA270, host_imageA270, img_sizeA90 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA270, dev_transparency_arr270, widthA90, heightA90, channelsA90);
	cudaDeviceSynchronize();
	int* transparency_arr270 = new int[widthA90 * heightA90];
	cudaMemcpy(transparency_arr270, dev_transparency_arr270, widthA90 * heightA90 * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr180;
	cudaMalloc((void**)&dev_transparency_arr180, widthA * heightA * sizeof(int));
	cudaMalloc((void**)&dev_imageA180, img_sizeA * sizeof(uint8_t));
	cudaMemcpy(dev_imageA180, host_imageA180, img_sizeA * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA180, dev_transparency_arr180, widthA, heightA, channelsA);
	cudaDeviceSynchronize();
	int* transparency_arr180 = new int[widthA * heightA];
	cudaMemcpy(transparency_arr180, dev_transparency_arr180, widthA * heightA * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr45;
	cudaMalloc((void**)&dev_transparency_arr45, widthA45 * heightA45 * sizeof(int));
	cudaMalloc((void**)&dev_imageA45, img_sizeA45 * sizeof(uint8_t));
	cudaMemcpy(dev_imageA45, host_imageA45, img_sizeA45 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA45, dev_transparency_arr45, widthA45, heightA45, channelsA45);
	cudaDeviceSynchronize();
	int* transparency_arr45 = new int[widthA45 * heightA45];
	cudaMemcpy(transparency_arr45, dev_transparency_arr45, widthA45 * heightA45 * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr135;
	cudaMalloc((void**)&dev_transparency_arr135, widthA45 * heightA45 * sizeof(int));
	cudaMalloc((void**)&dev_imageA135, img_sizeA45 * sizeof(uint8_t));
	cudaMemcpy(dev_imageA135, host_imageA135, img_sizeA45 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA135, dev_transparency_arr135, widthA45, heightA45, channelsA45);
	cudaDeviceSynchronize();
	int* transparency_arr135 = new int[widthA45 * heightA45];
	cudaMemcpy(transparency_arr135, dev_transparency_arr135, widthA45 * heightA45 * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr225;
	cudaMalloc((void**)&dev_transparency_arr225, widthA45 * heightA45 * sizeof(int));
	cudaMalloc((void**)&dev_imageA225, img_sizeA45 * sizeof(uint8_t));
	cudaMemcpy(dev_imageA225, host_imageA225, img_sizeA45 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA225, dev_transparency_arr225, widthA45, heightA45, channelsA45);
	cudaDeviceSynchronize();
	int* transparency_arr225 = new int[widthA45 * heightA45];
	cudaMemcpy(transparency_arr225, dev_transparency_arr225, widthA45 * heightA45 * sizeof(int), cudaMemcpyDeviceToHost);

	int* dev_transparency_arr315;
	cudaMalloc((void**)&dev_transparency_arr315, widthA45* heightA45 * sizeof(int));
	cudaMalloc((void**)&dev_imageA315, img_sizeA45 * sizeof(uint8_t));
	cudaMemcpy(dev_imageA315, host_imageA315, img_sizeA45 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	transparencyPixels << <gridSizeA, blockSize >> > (dev_imageA315, dev_transparency_arr315, widthA45, heightA45, channelsA45);
	cudaDeviceSynchronize();
	int* transparency_arr315 = new int[widthA45 * heightA45];
	cudaMemcpy(transparency_arr315, dev_transparency_arr315, widthA45* heightA45 * sizeof(int), cudaMemcpyDeviceToHost);




	//sprawdzenie i matching templatu ze zdjeciem

	int* dev_sample_check;
	int* dev_sample_check90;
	int* dev_sample_check270;
	int* dev_sample_check180;
	int* dev_sample_check45;
	int* dev_sample_check135;
	int* dev_sample_check225;
	int* dev_sample_check315;

	
	int treshold = int(0.97 * widthA * heightA);  // to ma sie zmieniac w zaleznosci od ilosci pixeli naszego sampla, cyli jezeli litera bedzie 100x 100 pixeli to np chcemy sprawdzic czy zgadza sie 90% czyli 0,9*100*100 pixeli jesli tak to wykryte
	int treshold45 = int(0.97 * widthA45 * heightA45);

	
	cudaMalloc((void**)&dev_image, width* height * 3 * sizeof(uint8_t));
	
	cudaMalloc((void**)&dev_sample_check, (width - widthA)* (height - heightA) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check90, (width - widthA90) * (height - heightA90) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check270, (width - widthA90) * (height - heightA90) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check180, (width - widthA) * (height - heightA) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check45, (width - widthA45)* (height - heightA45) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check135, (width - widthA45)* (height - heightA45) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check225, (width - widthA45)* (height - heightA45) * sizeof(int));
	cudaMalloc((void**)&dev_sample_check315, (width - widthA45)* (height - heightA45) * sizeof(int));


	cudaMemcpy(dev_image, host_image, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA, dev_sample_check, dev_transparency_arr, width, height, widthA, heightA, treshold);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA90, dev_sample_check90, dev_transparency_arr90, width, height, widthA90, heightA90, treshold);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA270, dev_sample_check270, dev_transparency_arr270, width, height, widthA90, heightA90, treshold);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA180, dev_sample_check180, dev_transparency_arr180, width, height, widthA, heightA, treshold);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA45, dev_sample_check45, dev_transparency_arr45, width, height, widthA45, heightA45, treshold45);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA135, dev_sample_check135, dev_transparency_arr135, width, height, widthA45, heightA45, treshold45);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA225, dev_sample_check225, dev_transparency_arr225, width, height, widthA45, heightA45, treshold45);
	subtractImagesKernel << <gridSize, blockSize >> > (dev_image, dev_imageA315, dev_sample_check315, dev_transparency_arr315, width, height, widthA45, heightA45, treshold45);


	cudaDeviceSynchronize();

	int* sample_check90 = new int[(width - widthA90) * (height - heightA90)];
	int* sample_check270 = new int[(width - widthA90) * (height - heightA90)];
	int* sample_check180 = new int[(width - widthA) * (height - heightA)];
	int* sample_check45 = new int[(width - widthA45) * (height - heightA45)];
	int* sample_check135 = new int[(width - widthA45) * (height - heightA45)];
	int* sample_check225 = new int[(width - widthA45) * (height - heightA45)];
	int* sample_check315 = new int[(width - widthA45) * (height - heightA45)];
	int* sample_check = new int[(width - widthA) * (height - heightA)];


	cudaMemcpy(sample_check, dev_sample_check, (width - widthA)* (height - heightA) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check90, dev_sample_check90, (width - widthA90) * (height - heightA90) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check270, dev_sample_check270, (width - widthA90) * (height - heightA90) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check180, dev_sample_check180, (width - widthA) * (height - heightA) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check45, dev_sample_check45, (width - widthA45)* (height - heightA45) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check135, dev_sample_check135, (width - widthA45)* (height - heightA45) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check225, dev_sample_check225, (width - widthA45)* (height - heightA45) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_check315, dev_sample_check315, (width - widthA45)* (height - heightA45) * sizeof(int), cudaMemcpyDeviceToHost);



	int samples = 0;
	for (int i = 0; i < (width - widthA) * (height - heightA); i++) {
		if (sample_check[i] == 1 || sample_check180[i] == 1) {
			samples++;
		}
	}
	for (int i = 0; i < (width - widthA90) * (height - heightA90); i++) {
		if (sample_check90[i] == 1 || sample_check270[i] == 1) {
			samples++;
		}
	}
	for (int i = 0; i < (width - widthA45) * (height - heightA45); i++) {
		if (sample_check45[i] == 1 || sample_check135[i] == 1 || sample_check225[i] == 1 || sample_check315[i] == 1) {
			samples++;
		}
	}




	//szukanie rogu
	int* dev_top_left;

	int* global_counter;
	cudaMalloc(&global_counter, sizeof(int));
	cudaMemset(global_counter, 0, sizeof(int)); // Zainicjuj licznik na 0
	cudaMalloc((void**)&dev_top_left, samples * 2 * sizeof(int));
	find_top_left << <gridSize, blockSize >> > (width, height, widthA, heightA, dev_sample_check, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA90, heightA90, dev_sample_check90, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA90, heightA90, dev_sample_check270, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA, heightA, dev_sample_check180, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA45, heightA45, dev_sample_check45, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA45, heightA45, dev_sample_check135, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA45, heightA45, dev_sample_check225, dev_top_left, global_counter);
	find_top_left << <gridSize, blockSize >> > (width, height, widthA45, heightA45, dev_sample_check315, dev_top_left, global_counter);


	cudaDeviceSynchronize();
	int* host_top_left = new int[samples * 2];
	
	cudaMemcpy(host_top_left, dev_top_left, samples * 2 * sizeof(int), cudaMemcpyDeviceToHost);

	//otaczanie bounding boxem
	uint8_t* dev_with_boundingBox;

	cudaMalloc((void**)&dev_with_boundingBox, width * height * 3 * sizeof(uint8_t));
	cudaMemcpy(dev_image, host_image, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	bounding_box << <gridSizeBB, blockSize >> > (dev_image, dev_with_boundingBox, width, height, widthA, heightA, dev_sample_check, dev_top_left, samples);
	cudaDeviceSynchronize();

	uint8_t* host_with_boundingBox = new uint8_t[width * height * 3];

	cudaMemcpy(host_with_boundingBox, dev_with_boundingBox, width * height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	stbi_write_jpg("boundingBox1.jpg", width, height, channels, host_with_boundingBox, 100);



	// Zwolnienie pamięci GPU
	cudaFree(dev_transparency_arr);
	cudaFree(dev_imageA);
	cudaFree(dev_image);
	cudaFree(dev_sample_check);
	cudaFree(dev_one);
	cudaFree(dev_three);
	cudaFree(dev_rotated90d_image);
	cudaFree(dev_rotated45d_image);
	cudaFree(dev_rotated135d_image);
	cudaFree(dev_rotated180d_image);
	cudaFree(dev_rotated225d_image);
	cudaFree(dev_rotated270d_image);
	cudaFree(dev_rotated315d_image);
	cudaFree(dev_increase_image);
	cudaFree(dev_with_boundingBox);
	cudaFree(global_counter);
	cudaFree(dev_top_left);
	cudaFree(dev_transparency_arr90);
	cudaFree(dev_imageA90);
	cudaFree(dev_sample_check90);
	cudaFree(dev_transparency_arr270);
	cudaFree(dev_imageA270);
	cudaFree(dev_sample_check270);
	cudaFree(dev_transparency_arr45);
	cudaFree(dev_imageA45);
	cudaFree(dev_sample_check45);
	cudaFree(dev_transparency_arr135);
	cudaFree(dev_imageA135);
	cudaFree(dev_sample_check135);
	cudaFree(dev_transparency_arr180);
	cudaFree(dev_imageA180);
	cudaFree(dev_sample_check180);
	cudaFree(dev_transparency_arr225);
	cudaFree(dev_imageA225);
	cudaFree(dev_sample_check225);
	cudaFree(dev_transparency_arr315);
	cudaFree(dev_imageA315);
	cudaFree(dev_sample_check315);

	// Zwolnienie pamięci hosta
	stbi_image_free(host_image);
	stbi_image_free(host_imageA);
	delete[] transparency_arr;
	delete[] sample_check;
	delete[] host_three_dimension;
	delete[] host_one_dimension;
	delete[] host_rotated90d_image;
	delete[] host_rotated45d_image;
	delete[] host_rotated135d_image;
	delete[] host_rotated180d_image;
	delete[] host_rotated225d_image;
	delete[] host_rotated270d_image;
	delete[] host_rotated315d_image;
	delete[] host_increase_image;
	delete[] host_with_boundingBox;
	delete[] host_top_left;
	delete[] transparency_arr45;
	delete[] transparency_arr90;
	delete[] transparency_arr135;
	delete[] transparency_arr180;
	delete[] transparency_arr225;
	delete[] transparency_arr270;
	delete[] transparency_arr315;

	cout << endl;
	cout << endl;
	cout << widthA << ' ' << heightA << endl << width << ' ' << height << endl;
	cout << endl;

	return 0;
}



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


__global__ void rotate90degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int newIndex = y + (width - x - 1) * height;
		int oldIndex = y * width + x;
		dev_output[newIndex] = dev_input[oldIndex];
	}
}

__global__ void rotate270degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int newIndex = (height - y - 1) + x * height;
		int oldIndex = y * width + x;
		dev_output[newIndex] = dev_input[oldIndex];
	}
}
__global__ void rotate180degree(uint8_t* dev_input, uint8_t* dev_output, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int newIndex = (height - y - 1) * width + (width - x - 1);
		int oldIndex = y * width + x;
		dev_output[newIndex] = dev_input[oldIndex];
	}
}



__global__ void rotate45degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {

		float centerX = width / 2.0;
		float centerY = height / 2.0;

		float newX = x - newWidth / 2.0;
		float newY = y - newHeight / 2.0;

		float theta = -M_PI / 4.0;
		float oldX = cosf(theta) * newX - sinf(theta) * newY + centerX;
		float oldY = sinf(theta) * newX + cosf(theta) * newY + centerY;

		if (oldX >= 0 && oldX < width && oldY >= 0 && oldY < height) {
			int oldIndex = (int)roundf(oldY) * width + (int)roundf(oldX);
			output[y * newWidth + x] = input[oldIndex];
		}
	}
}

__global__ void rotate135degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {

		float centerX = width / 2.0;
		float centerY = height / 2.0;

		float newX = x - newWidth / 2.0;
		float newY = y - newHeight / 2.0;

		float theta = -3.0f * M_PI / 4.0f;
		float oldX = cosf(theta) * newX - sinf(theta) * newY + centerX;
		float oldY = sinf(theta) * newX + cosf(theta) * newY + centerY;

		if (oldX >= 0 && oldX < width && oldY >= 0 && oldY < height) {
			int oldIndex = (int)roundf(oldY) * width + (int)roundf(oldX);
			output[y * newWidth + x] = input[oldIndex];
		}
		else {
			output[y * newWidth + x] = 255;
		}
	}
}

__global__ void rotate225degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {
		float centerX = width / 2.0;
		float centerY = height / 2.0;

		float newX = x - newWidth / 2.0;
		float newY = y - newHeight / 2.0;

		float theta = -3.0f * M_PI / 4.0f;
		float oldX = cosf(theta) * newX + sinf(theta) * newY + centerX;
		float oldY = -sinf(theta) * newX + cosf(theta) * newY + centerY;

		if (oldX >= 0 && oldX < width && oldY >= 0 && oldY < height) {
			int oldIndex = (int)roundf(oldY) * width + (int)roundf(oldX);
			output[y * newWidth + x] = input[oldIndex];
		}
		else {
			output[y * newWidth + x] = 255;
		}
	}
}



__global__ void rotate315degree(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {

		float centerX = width / 2.0;
		float centerY = height / 2.0;

		float newX = x - newWidth / 2.0;
		float newY = y - newHeight / 2.0;

		float theta = -M_PI / 4.0;
		float oldX = cosf(theta) * newX + sinf(theta) * newY + centerX;
		float oldY = -sinf(theta) * newX + cosf(theta) * newY + centerY;

		if (oldX >= 0 && oldX < width && oldY >= 0 && oldY < height) {
			int oldIndex = (int)roundf(oldY) * width + (int)roundf(oldX);
			output[y * newWidth + x] = input[oldIndex];
		}
	}
}


__global__ void increase_letters(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {
		int srcX = x / 1.5;
		int srcY = y / 1.5;

		if (srcX < width && srcY < height) {
			output[y * newWidth + x] = input[srcY * width + srcX];
		}
	}

}

__global__ void decrease_letters(uint8_t* input, uint8_t* output, int width, int height, int newWidth, int newHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < newWidth && y < newHeight) {
		int srcX = x * 1.5;
		int srcY = y * 1.5;

		if (srcX < width && srcY < height) {
			output[y * newWidth + x] = input[srcY * width + srcX];
		}
	}
}

__global__ void bounding_box(uint8_t* input, uint8_t* output, int width, int height, int widthA, int heightA, int* sample_check, int* top_left, int samples) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = (y * width + x) * 3; // Indeks dla formatu RGB

		output[idx] = input[idx];
		output[idx + 1] = input[idx + 1];
		output[idx + 2] = input[idx + 2];

		for (int i = 0; i < samples; i++) {
			int box_x = top_left[i * 2];
			int box_y = top_left[i * 2 + 1];
			//atomicMax(a, i);
			// Sprawdzanie, czy piksel znajduje się na krawędziach któregokolwiek bounding boxa
			if (x >= box_x && x <= box_x + heightA &&
				y >= box_y && y <= box_y + heightA &&
				(x == box_x || x == box_x + heightA || y == box_y || y == box_y + heightA)) {
				// Rysuj krawędzie prostokąta
				output[idx] = 0;     // Czerwony
				output[idx + 1] = 255;   // Zielony
				output[idx + 2] = 0;   // Niebieski
				break; // Zakończ pętlę, ponieważ piksel jest już na krawędzi bounding boxa
			}
		}

	}

}

__global__ void find_top_left(int width, int height, int widthA, int heightA, int* sample_check, int* top_left, int* global_counter) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x <= (width - widthA) && y <= (height - heightA)) {
		if (sample_check[y * (width - widthA) + x] == 1) {
			int index = atomicAdd(global_counter, 1);				//podkreśla, ale działa nie wiem co mu się nie podoba
			top_left[index * 2] = x;
			top_left[index * 2 + 1] = y;
			//printf("%d, %d \n", top_left[index * 2],top_left[index * 2 + 1] = y);
		}
	}

}

__global__ void find_top_left90(int width, int height, int widthA90, int heightA90, int* sample_check, int* top_left, int* global_counter) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x <= (width - widthA90) && y <= (height - heightA90)) {
		if (sample_check[y * (width - widthA90) + x] == 1) {
			int index = atomicAdd(global_counter, 1);				//podkreśla, ale działa nie wiem co mu się nie podoba
			top_left[index * 2] = x;
			top_left[index * 2 + 1] = y;
			//printf("%d, %d \n", top_left[index * 2],top_left[index * 2 + 1] = y);
		}
	}

}