#include <iostream>

#include "Image.hpp"
#include "FlipMetric.hpp"

int main() {
    std::vector<unsigned char> imgData;
    auto img1 = Image::readPNG("input_mj.png");
    auto img2 = Image::readPNG("best.png");

    FlipMetric metric(img1->getData(), img1->getWidth(), img1->getHeight());

    std::cout << "1: " << metric.compareHost(img1->getData()) << std::endl;
    imgData = std::vector<unsigned char>(img1->getWidth() * img1->getHeight() * 3);

    std::cout << "2: " << metric.compareHost(img2->getData()) << std::endl;
    imgData = std::vector<unsigned char>(img2->getWidth() * img2->getHeight() * 3);

    return 0;
}
