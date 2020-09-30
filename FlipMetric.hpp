#pragma once

#include <memory>

class FlipMetricImpl;

class FlipMetric {
private:
    std::shared_ptr<FlipMetricImpl> m_impl;

public:
    FlipMetric(const unsigned char* reference, unsigned int width, unsigned int height, float ppd = 67.0206f);
    ~FlipMetric();

    float compareDevice(const unsigned char* image);
    float compareHost(const unsigned char* image);
};
