#pragma once

#include <thrust/device_vector.h>

#include "Color.hpp"

class FlipMetricImpl {
private:
    typedef thrust::device_vector<Color> ColorVec;

    float m_ppd;
    unsigned int m_imageWidth;
    unsigned int m_imageHeight;
    unsigned int m_colorFilterWidth;
    unsigned int m_featureFilterWidth;

    ColorVec m_referencePixels;
    ColorVec m_referenceGrayPixels;
    ColorVec m_testPixels;
    ColorVec m_testGrayPixels;
    ColorVec m_colorDifference;
    ColorVec m_featureDifference;

    ColorVec m_colorFilter;
    ColorVec m_edgesFilter;
    ColorVec m_pointsFilter;

    ColorVec m_edgesReference;
    ColorVec m_edgesTest;
    ColorVec m_pointsReference;
    ColorVec m_pointsTest;

    thrust::device_vector<float> m_flip;
    thrust::device_vector<float> m_histogram;
    thrust::device_vector<float> m_histogramSeq;

    float gaussian(const float x, const float y, const float sigma);
    void sRGB2YCxCz(const unsigned char* input, ColorVec& output);
    void YCxCz2Gray(const ColorVec& input, ColorVec& output);
    void YCxCz2CIELab(const ColorVec& input, ColorVec& output);
    void huntAdjustment(const ColorVec& input, ColorVec& output);
    void normalize(const ColorVec& input, ColorVec& output, const Color& total);
    void generateSpatialFilter(ColorVec& output, unsigned int width, float radius, float deltaX);
    void convolve(const ColorVec& image, unsigned int imageWidth, unsigned int imageHeight, const ColorVec& filter, unsigned int filterWidth, unsigned int filterHeight, ColorVec& output);
    void computeColorDifference(const ColorVec& reference, const ColorVec& test, ColorVec& output);
    void computeFeatureDifference(const ColorVec& edgesReference, const ColorVec& edgesTest, const ColorVec& pointsReference, const ColorVec& pointsTest, ColorVec& output);
    void computeFlipError(const ColorVec& colorDiff, const ColorVec& featureDiff, thrust::device_vector<float>& output);

    void createColorFilter();
    void createDetectionFilter(ColorVec& output, float stdDev, float radius, int width, bool pointDetector);
    void createDetectionFilters();
    void preprocess(ColorVec& image, ColorVec& imageGray, const ColorVec& colorFilter);

public:
    FlipMetricImpl(const unsigned char* image, unsigned int width, unsigned int height, float ppd);
    ~FlipMetricImpl();

    float compareDevice(const unsigned char* image);
    float compareHost(const unsigned char* image);
};
