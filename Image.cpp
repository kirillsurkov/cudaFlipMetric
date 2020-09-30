#include "Image.hpp"
#include "svpng.inc"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <thread>
#include <mutex>
#include <png.h>

void Image::brightnessContrast(float brightness, float contrast, unsigned char& r, unsigned char& g, unsigned char& b) {
    float factor = (259.0f * (contrast + 255)) / (255.0f * (259 - contrast));
    r = std::max(std::min(brightness + factor * (r - 128) + 128, 255.0f), 0.0f);
    g = std::max(std::min(brightness + factor * (g - 128) + 128, 255.0f), 0.0f);
    b = std::max(std::min(brightness + factor * (b - 128) + 128, 255.0f), 0.0f);
}

void Image::savePNG(const std::string& filename, std::shared_ptr<Image> image) {
    std::vector<unsigned char> data(image->m_width * image->m_height * 3);
    for (unsigned int y = 0; y < image->m_height; y++) {
        auto pixelInput = image->m_dataPtr + y * image->m_stride;
        auto pixelOutput = data.data() + y * image->m_width * 3;
        for (unsigned int x = 0; x < image->m_width; x++) {
            pixelOutput[x * 3 + 0] = pixelInput[x * 3 + 0];
            pixelOutput[x * 3 + 1] = pixelInput[x * 3 + 1];
            pixelOutput[x * 3 + 2] = pixelInput[x * 3 + 2];
        }
    }
    FILE* file = fopen(filename.c_str(), "wb");
    svpng(file, image->m_width, image->m_height, data.data(), 0);
    fclose(file);
}

std::shared_ptr<Image> Image::readPNG(const std::string& filename) {
    int width;
    int height;
    unsigned char header[8];    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp)
        throw std::runtime_error("[read_png_file] File '" + filename + "' could not be opened for reading");
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
        throw std::runtime_error("[read_png_file] File '" + filename + "' is not recognized as a PNG file");


    /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        throw std::runtime_error("[read_png_file] png_create_read_struct failed");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        throw std::runtime_error("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    unsigned int channels = (color_type == PNG_COLOR_TYPE_RGB ? 3 : 4);

    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
        throw std::runtime_error("[read_png_file] Error during read_image");

    png_bytep* row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

    png_read_image(png_ptr, row_pointers);

    std::vector<unsigned char> data;
    for (int y = 0; y < height; y++) {
        png_byte* row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_byte* pixel = &row[x * channels];
            data.push_back(pixel[0]);
            data.push_back(pixel[1]);
            data.push_back(pixel[2]);
        }
        free(row);
    }

    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

    fclose(fp);

    return std::make_shared<Image>(width, height, width * 3, std::move(data));
}

Image::Image(unsigned int width, unsigned int height, unsigned int stride, std::vector<unsigned char>&& data) : m_width(width), m_height(height), m_stride(stride), m_data(data), m_dataPtr(m_data.data()) {
}

Image::Image(unsigned int width, unsigned int height, unsigned int stride, unsigned char* data) : m_width(width), m_height(height), m_stride(stride), m_dataPtr(data) {
}

unsigned int Image::getWidth() const {
    return m_width;
}

unsigned int Image::getHeight() const {
    return m_height;
}

unsigned int Image::getStride() const {
    return m_stride;
}

const unsigned char* Image::getData() const {
    return m_dataPtr;
}

unsigned char* Image::getData() {
    return m_dataPtr;
}

void Image::applyBC(float brightness, float contrast) {
    unsigned int totalPixels = m_width * m_height;
    unsigned char* thisData = m_data.data();
    for (unsigned int i = 0; i < totalPixels; i++) {
        unsigned int x = i % m_width;
        unsigned int y = i / m_height;
        unsigned char* pixel = thisData + (y * m_width + x) * 3;
        brightnessContrast(brightness, contrast, pixel[0], pixel[1], pixel[2]);
    }
}
