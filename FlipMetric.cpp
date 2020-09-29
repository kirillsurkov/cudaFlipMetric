#include "FlipMetric.hpp"
#include "FlipMetricImpl.cuh"

FlipMetric::FlipMetric(const unsigned char* reference, unsigned int width, unsigned int height, float ppd) {
    m_impl = std::make_shared<FlipMetricImpl>(reference, width, height, ppd);
}

FlipMetric::~FlipMetric() {
}

float FlipMetric::compareDevice(const unsigned char* image) {
    return m_impl->compareDevice(image);
}

float FlipMetric::compareHost(const unsigned char* image) {
    return m_impl->compareHost(image);
}
