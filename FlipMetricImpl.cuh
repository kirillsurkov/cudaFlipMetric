#pragma once

#include <thrust/device_vector.h>
#include <cudnn.h>

#include "Color.hpp"

class FlipMetricImpl {
private:
    typedef thrust::device_vector<Color> ColorVec;

    class Filter {
    private:
        ColorVec m_hwc;
        thrust::device_vector<float> m_chw;
        cudnnFilterDescriptor_t m_filterDescriptor;
        cudnnConvolutionDescriptor_t m_convolutionDescriptor;
        cudnnConvolutionFwdAlgoPerf_t m_convolutionAlgorithm;
        void* m_workspace;
        size_t m_workspaceSize;

    public:
        Filter(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t inputDescriptor, cudnnTensorDescriptor_t outputDescriptor, unsigned int width, unsigned int height);
        ~Filter();

        cudnnFilterDescriptor_t getFilterDescriptor();
        cudnnConvolutionDescriptor_t getConvolutionDescriptor();
        cudnnConvolutionFwdAlgo_t getConvolutionAlgorithm();
        void* getWorkspace();
        size_t getWorkspaceSize();
        ColorVec& getDataHWC();
        thrust::device_vector<float>& getDataCHW();

        void toCHW();
    };

    float m_ppd;
    unsigned int m_imageWidth;
    unsigned int m_imageHeight;

    std::shared_ptr<Filter> m_colorFilter;
    std::shared_ptr<Filter> m_edgesFilter;
    std::shared_ptr<Filter> m_pointsFilter;

    ColorVec m_referencePixels;
    ColorVec m_referenceGrayPixels;
    ColorVec m_testPixels;
    ColorVec m_testGrayPixels;
    ColorVec m_colorDifference;
    ColorVec m_featureDifference;

    ColorVec m_edgesReference;
    ColorVec m_edgesTest;
    ColorVec m_pointsReference;
    ColorVec m_pointsTest;

    thrust::device_vector<float> m_chwInput;
    thrust::device_vector<float> m_chwOutput;
    thrust::device_vector<float> m_flip;
    thrust::device_vector<float> m_histogram;
    thrust::device_vector<float> m_histogramSeq;

    cudnnHandle_t m_cudnnHandle;
    cudnnTensorDescriptor_t m_inputDescriptor;
    cudnnTensorDescriptor_t m_outputDescriptor;

    float gaussian(const float x, const float y, const float sigma);
    void sRGB2YCxCz(const unsigned char* input, ColorVec& output);
    void YCxCz2Gray(const ColorVec& input, ColorVec& output);
    void YCxCz2CIELab(const ColorVec& input, ColorVec& output);
    void huntAdjustment(const ColorVec& input, ColorVec& output);
    void normalize(const ColorVec& input, ColorVec& output, const Color& total);
    void generateSpatialFilter(ColorVec& output, unsigned int width, float radius, float deltaX);
    void convolve(const std::shared_ptr<Filter>& filter, const ColorVec& image, ColorVec& output);
    void computeColorDifference(const ColorVec& reference, const ColorVec& test, ColorVec& output);
    void computeFeatureDifference(const ColorVec& edgesReference, const ColorVec& edgesTest, const ColorVec& pointsReference, const ColorVec& pointsTest, ColorVec& output);
    void computeFlipError(const ColorVec& colorDiff, const ColorVec& featureDiff, thrust::device_vector<float>& output);

    void createColorFilter();
    void createDetectionFilter(ColorVec& output, float stdDev, float radius, int width, bool pointDetector);
    void createDetectionFilters();
    void preprocess(ColorVec& image, ColorVec& imageGray);

public:
    FlipMetricImpl(const unsigned char* image, unsigned int width, unsigned int height, float ppd);
    ~FlipMetricImpl();

    float compareDevice(const unsigned char* image);
    float compareHost(const unsigned char* image);
};
