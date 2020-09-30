#pragma once

#include <string>
#include <vector>
#include <memory>

class Image : public std::enable_shared_from_this<Image> {
private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_stride;
    std::vector<unsigned char> m_data;
    unsigned char* m_dataPtr;

    static void brightnessContrast(float brightness, float contrast, unsigned char& r, unsigned char& g, unsigned char& b);

public:
    static void savePNG(const std::string& filename, std::shared_ptr<Image> image);
    static std::shared_ptr<Image> readPNG(const std::string& filename);

    Image(unsigned int width, unsigned int height, unsigned int stride, std::vector<unsigned char>&& data);
    Image(unsigned int width, unsigned int height, unsigned int stride, unsigned char* data);

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getStride() const;
    const unsigned char* getData() const;
    unsigned char* getData();

    void applyBC(float brightness, float contrast);
    void applyC(float contrast);
};
