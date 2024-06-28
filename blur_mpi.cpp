#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "lodepng.h"

struct Pixel {
    unsigned char r, g, b, a;
};

void applyGaussianBlur(std::vector<Pixel>& image, int width, int height,int rank, bool testing = false,  int kernelSize = 11,  float sigma = 10.0f) {
    std::vector<Pixel> tempImage = image;
    float kernelSum = 0.0f;
    std::vector<float> kernel(kernelSize * kernelSize);

    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++) {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            float value = exp(exponent) / (2 * M_PI * sigma * sigma);
            kernel[(y + kernelSize / 2) * kernelSize + (x + kernelSize / 2)] = value;
            kernelSum += value;
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= kernelSum;
    }

    for (int y = kernelSize / 2; y < height - kernelSize / 2; y++) {
        for (int x = kernelSize / 2; x < width - kernelSize / 2; x++) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumA = 0.0f;
            for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++) {
                for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++) {
                    int index = (y + ky) * width + (x + kx);
                    sumR += tempImage[index].r * kernel[(ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2)];
                    sumG += tempImage[index].g * kernel[(ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2)];
                    sumB += tempImage[index].b * kernel[(ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2)];
                    sumA += tempImage[index].a * kernel[(ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2)];
                }
            }
            int index = y * width + x;
            if(testing){
                // Apply tint based on rank, cycling through a few distinct colors
                unsigned char tint = rank * 30 % 256;
                image[index].r = std::min(static_cast<int>(sumR) + (rank % 3 == 0 ? tint : 0), 255);
                image[index].g = std::min(static_cast<int>(sumG) + (rank % 3 == 1 ? tint : 0), 255);
                image[index].b = std::min(static_cast<int>(sumB) + (rank % 3 == 2 ? tint : 0), 255);
                image[index].a = static_cast<unsigned char>(sumA);  // Preserving alpha
            }else{
                image[index].r = static_cast<unsigned char>(sumR);
                image[index].g = static_cast<unsigned char>(sumG);
                image[index].b = static_cast<unsigned char>(sumB);
                image[index].a = static_cast<unsigned char>(sumA);
            }
      
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_image.png> <output_image.png>\n";
        }
        MPI_Finalize();
        return 1;
    }

    char* input_filename = argv[1];
    char* output_filename = argv[2];

    std::vector<unsigned char> image;
    unsigned width, height;
    int error;

    if (rank == 0) {
        error = lodepng::decode(image, width, height, input_filename);
        if (error) {
            std::cerr << "Error reading PNG file: " << lodepng_error_text(error) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    unsigned dimensions[2];
    if (rank == 0) {
        dimensions[0] = width;
        dimensions[1] = height;
    }
    
    MPI_Bcast(dimensions, 2, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        width = dimensions[0];
        height = dimensions[1];
    }

    

    int chunk_size = height / size;
    std::vector<unsigned char> local_image(width * chunk_size * 4);

    MPI_Scatter(image.data(), width * chunk_size * 4, MPI_UNSIGNED_CHAR, local_image.data(), width * chunk_size * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    std::vector<Pixel> local_pixels(chunk_size * width);
    for (int i = 0; i < local_pixels.size(); ++i) {
        local_pixels[i].r = local_image[i * 4];
        local_pixels[i].g = local_image[i * 4 + 1];
        local_pixels[i].b = local_image[i * 4 + 2];
        local_pixels[i].a = local_image[i * 4 + 3];
    }

    applyGaussianBlur(local_pixels, width, chunk_size,rank);

    for (int i = 0; i < local_pixels.size(); ++i) {
        local_image[i * 4] = local_pixels[i].r;
        local_image[i * 4 + 1] = local_pixels[i].g;
        local_image[i * 4 + 2] = local_pixels[i].b;
        local_image[i * 4 + 3] = local_pixels[i].a;
    }

    MPI_Gather(local_image.data(), width * chunk_size * 4, MPI_UNSIGNED_CHAR, image.data(), width * chunk_size * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        error = lodepng::encode(output_filename, image, width, height);
        if (error) {
            std::cerr << "Error writing PNG file: " << lodepng_error_text(error) << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Finalize();
    return 0;
}
