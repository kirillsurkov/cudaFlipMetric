#include "FlipMetricImpl.cuh"

#include <cmath>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

__device__ float __cuFlip_min(float x, float y) {
    return x < y ? x : y;
}

__device__ float __cuFlip_max(float x, float y) {
    return x > y ? x : y;
}

__device__ float __cuFlip_abs(float x) {
    return x < 0 ? -x : x;
}

__device__ float __cuFlip_HyAB(const Color& refPixel, const Color& testPixel) {
    float cityBlockDistanceL = fabsf(refPixel.x - testPixel.x);
    float euclideanDistanceAB = sqrtf((refPixel.y - testPixel.y) * (refPixel.y - testPixel.y) + (refPixel.z - testPixel.z) * (refPixel.z - testPixel.z));
    return cityBlockDistanceL + euclideanDistanceAB;
}

__device__ float __cuFlip_GaussSum(const float x2, const float a1, const float b1, const float a2, const float b2) {
    const float pi = float(M_PI);
    const float pi_sq = float(M_PI * M_PI);
    return a1 * sqrtf(pi / b1) * expf(-pi_sq * x2 / b1) + a2 * sqrtf(pi / b2) * expf(-pi_sq * x2 / b2);
}

__device__ float __cuFlip_sRGB2Linear(float sRGBColor) {
    if (sRGBColor <= 0.04045f) {
        return sRGBColor / 12.92f;
    } else {
        return powf((sRGBColor + 0.055f) / 1.055f, 2.4f);
    }
}

__device__ void __cuFlip_LinearRGB2XYZ(float& r, float& g, float& b) {
    const float a11 = 10135552.0f / 24577794.0f;
    const float a12 = 8788810.0f / 24577794.0f;
    const float a13 = 4435075.0f / 24577794.0f;
    const float a21 = 2613072.0f / 12288897.0f;
    const float a22 = 8788810.0f / 12288897.0f;
    const float a23 = 887015.0f / 12288897.0f;
    const float a31 = 1425312.0f / 73733382.0f;
    const float a32 = 8788810.0f / 73733382.0f;
    const float a33 = 70074185.0f / 73733382.0f;
    float vR = r;
    float vG = g;
    float vB = b;
    r = a11 * vR + a12 * vG + a13 * vB;
    g = a21 * vR + a22 * vG + a23 * vB;
    b = a31 * vR + a32 * vG + a33 * vB;
}

__global__ void __cuFlip_sRGB2YCxCz(const unsigned char* input, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Color reference_illuminant = {0.950428545377181f, 1.0f, 1.088900370798128f};

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        float x = __cuFlip_sRGB2Linear(input[i * 3 + 0] / 255.0f);
        float y = __cuFlip_sRGB2Linear(input[i * 3 + 1] / 255.0f);
        float z = __cuFlip_sRGB2Linear(input[i * 3 + 2] / 255.0f);

        __cuFlip_LinearRGB2XYZ(x, y, z);

        x /= reference_illuminant.x;
        y /= reference_illuminant.y;
        z /= reference_illuminant.z;

        Color& out = output[i];
        out.x = 116.0f * y - 16.0f;
        out.y = 500.0f * (x - y);
        out.z = 200.0f * (y - z);
    }
}

__global__ void __cuFlip_YCxCz2Gray(const Color* input, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        const Color& pixel = input[i];

        float c = (pixel.x + 16.0f) / 116.0f;

        Color& out = output[i];
        out.x = c;
        out.y = c;
        out.z = 0.0f;
    }
}

__global__ void __cuFlip_YCxCz2CIELab(const Color* input, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Color reference_illuminant = {0.950428545377181f, 1.0f, 1.088900370798128f};

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        const Color& src = input[i];
        Color& out = output[i];
        Color YCxCz = src;

        const float Yy = (YCxCz.x + 16.0f) / 116.0f;
        const float Cx = YCxCz.y / 500.0f;
        const float Cz = YCxCz.z / 200.0f;
        out.x = Yy + Cx;
        out.y = Yy;
        out.z = Yy - Cz;
        out.x *= reference_illuminant.x;
        out.y *= reference_illuminant.y;
        out.z *= reference_illuminant.z;

        const float a11 = 3.241003232976358f;
        const float a12 = -1.537398969488785f;
        const float a13 = -0.498615881996363f;
        const float a21 = -0.969224252202516f;
        const float a22 = 1.875929983695176f;
        const float a23 = 0.041554226340085f;
        const float a31 = 0.055639419851975f;
        const float a32 = -0.204011206123910f;
        const float a33 = 1.057148977187533f;
        Color v = out;
        out.x = __cuFlip_min(__cuFlip_max(a11 * v.x + a12 * v.y + a13 * v.z, 0.0f), 1.0f);
        out.y = __cuFlip_min(__cuFlip_max(a21 * v.x + a22 * v.y + a23 * v.z, 0.0f), 1.0f);
        out.z = __cuFlip_min(__cuFlip_max(a31 * v.x + a32 * v.y + a33 * v.z, 0.0f), 1.0f);

        const float b11 = 10135552.0f / 24577794.0f;
        const float b12 = 8788810.0f / 24577794.0f;
        const float b13 = 4435075.0f / 24577794.0f;
        const float b21 = 2613072.0f / 12288897.0f;
        const float b22 = 8788810.0f / 12288897.0f;
        const float b23 = 887015.0f / 12288897.0f;
        const float b31 = 1425312.0f / 73733382.0f;
        const float b32 = 8788810.0f / 73733382.0f;
        const float b33 = 70074185.0f / 73733382.0f;
        v = out;
        out.x = __cuFlip_abs(b11 * v.x + b12 * v.y + b13 * v.z);
        out.y = __cuFlip_abs(b21 * v.x + b22 * v.y + b23 * v.z);
        out.z = __cuFlip_abs(b31 * v.x + b32 * v.y + b33 * v.z);

        Color xyz = out;
        xyz.x /= reference_illuminant.x;
        xyz.y /= reference_illuminant.y;
        xyz.z /= reference_illuminant.z;
        xyz.x = xyz.x > 0.008856 ? powf(xyz.x, 1.0f / 3.0f) : 7.787f * xyz.x + 16.0f / 116.0f;
        xyz.y = xyz.y > 0.008856 ? powf(xyz.y, 1.0f / 3.0f) : 7.787f * xyz.y + 16.0f / 116.0f;
        xyz.z = xyz.z > 0.008856 ? powf(xyz.z, 1.0f / 3.0f) : 7.787f * xyz.z + 16.0f / 116.0f;
        out.x = 116.0f * xyz.y - 16.0f;
        out.y = 500.0f * (xyz.x - xyz.y);
        out.z = 200.0f * (xyz.y - xyz.z);
    }
}

__global__ void __cuFlip_huntAdjustment(const Color* input, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        const Color& src = input[i];
        Color& out = output[i];

        out.y = 0.01f * src.x * src.y;
        out.z = 0.01f * src.x * src.z;
    }
}

__global__ void __cuFlip_normalize(const Color* input, Color* output, Color total, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        const Color& src = input[i];
        Color& out = output[i];
        out.x = src.x / total.x;
        out.y = src.y / total.y;
        out.z = src.z / total.z;
    }
}

__global__ void __cuFlip_generateSpatialFilter(Color* output, unsigned int width, float radius, float deltaX) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Color a1 = {1.0f, 1.0f, 34.1f};
    Color b1 = {0.0047f, 0.0053f, 0.04f };
    Color a2 = { 0.0f, 0.0f, 13.5f };
    Color b2 = { 1.0e-5f, 1.0e-5f, 0.025f };

    unsigned int pixelsCount = width * width;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        unsigned int x = i % width;
        unsigned int y = i / width;

        float iy = (y - radius) * deltaX;
        float ix = (x - radius) * deltaX;

        float dist2 = ix * ix + iy * iy;
        output[i] = Color{__cuFlip_GaussSum(dist2, a1.x, b1.x, a2.x, b2.x), __cuFlip_GaussSum(dist2, a1.y, b1.y, a2.y, b2.y), __cuFlip_GaussSum(dist2, a1.z, b1.z, a2.z, b2.z)};
    }
}

__global__ void __cuFlip_computeColorDifference(const Color* reference, const Color* test, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float gpc = 0.4f;
    const float gqc = 0.7f;
    const float gpt = 0.95f;
    const float cmax = 41.2761f;
    const float pccmax = gpc * cmax;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        // compute difference in HyAB
        Color refPixel = reference[i];
        Color testPixel = test[i];
        float error = __cuFlip_HyAB(refPixel, testPixel);

        error = powf(error, gqc);

        // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
        // while the rest are mapped to the range (gpt, 1]
        if (error < pccmax) {
            error *= gpt / pccmax;
        } else {
            error = gpt + ((error - pccmax) / (cmax - pccmax)) * (1.0f - gpt);
        }

        Color& out = output[i];
        out.x = error;
        out.y = 0.0f;
        out.z = 0.0f;
    }
}

__global__ void __cuFlip_computeFeatureDifference(const Color* edgesReference, const Color* edgesTest, const Color* pointsReference, const Color* pointsTest, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float gqf = 0.5f;
    const float normalizationFactor = 1.0f / sqrtf(2.0f);
    Color p;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        p = edgesReference[i];
        const float edgeValueRef = sqrtf(p.x * p.x + p.y * p.y);

        p = edgesTest[i];
        const float edgeValueTest = sqrtf(p.x * p.x + p.y * p.y);

        p = pointsReference[i];
        const float pointValueRef = sqrtf(p.x * p.x + p.y * p.y);

        p = pointsTest[i];
        const float pointValueTest = sqrtf(p.x * p.x + p.y * p.y);

        const float edgeDifference = __cuFlip_abs(edgeValueRef - edgeValueTest);
        const float pointDifference = __cuFlip_abs(pointValueRef - pointValueTest);

        const float featureDifference = pow(normalizationFactor * __cuFlip_max(edgeDifference, pointDifference), gqf);

        Color& out = output[i];
        out.x = featureDifference;
        out.y = 0.0f;
        out.z = 0.0f;
    }
}

__global__ void __cuFlip_computeFlipError(const Color* colorDifference, const Color* featureDifference, float* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        const float cdiff = colorDifference[i].x;
        const float fdiff = featureDifference[i].x;
        const float errorFLIP = std::pow(cdiff, 1.0f - fdiff);

        output[i] = errorFLIP;
    }
}

__global__ void __cuFlip_hwc2chw(const Color* input, float* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        output[i + pixelsCount * 0] = input[i].x;
        output[i + pixelsCount * 1] = input[i].y;
        output[i + pixelsCount * 2] = input[i].z;
    }
}

__global__ void __cuFlip_chw2hwc(const float* input, Color* output, unsigned int pixelsCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < pixelsCount; i += stride) {
        output[i].x = input[i + pixelsCount * 0];
        output[i].y = input[i + pixelsCount * 1];
        output[i].z = input[i + pixelsCount * 2];
    }
}

FlipMetricImpl::Filter::Filter(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t inputDescriptor, cudnnTensorDescriptor_t outputDescriptor, unsigned int width, unsigned int height) {
    m_hwc.resize(width * height);
    m_chw.resize(width * height * 3);

    cudnnCreateFilterDescriptor(&m_filterDescriptor);
    cudnnSetFilter4dDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 1, height, width);

    cudnnCreateConvolutionDescriptor(&m_convolutionDescriptor);
    cudnnSetConvolution2dDescriptor(m_convolutionDescriptor, height / 2, width / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int count;
    cudnnFindConvolutionForwardAlgorithm(cudnnHandle, inputDescriptor, m_filterDescriptor, m_convolutionDescriptor, outputDescriptor, 1, &count, &m_convolutionAlgorithm);

    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDescriptor, m_filterDescriptor, m_convolutionDescriptor, outputDescriptor, m_convolutionAlgorithm.algo, &m_workspaceSize);

    cudaMalloc(&m_workspace, m_workspaceSize);
}

FlipMetricImpl::Filter::~Filter() {
    cudnnDestroyFilterDescriptor(m_filterDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convolutionDescriptor);
    cudaFree(&m_workspace);
}

cudnnFilterDescriptor_t FlipMetricImpl::Filter::getFilterDescriptor() {
    return m_filterDescriptor;
}

cudnnConvolutionDescriptor_t FlipMetricImpl::Filter::getConvolutionDescriptor() {
    return m_convolutionDescriptor;
}

cudnnConvolutionFwdAlgo_t FlipMetricImpl::Filter::getConvolutionAlgorithm() {
    return m_convolutionAlgorithm.algo;
}

void* FlipMetricImpl::Filter::getWorkspace() {
    return m_workspace;
}

size_t FlipMetricImpl::Filter::getWorkspaceSize() {
    return m_workspaceSize;
}

FlipMetricImpl::ColorVec& FlipMetricImpl::Filter::getDataHWC() {
    return m_hwc;
}

thrust::device_vector<float>& FlipMetricImpl::Filter::getDataCHW() {
    return m_chw;
}

void FlipMetricImpl::Filter::toCHW() {
    __cuFlip_hwc2chw<<<128, 4>>>(thrust::raw_pointer_cast(m_hwc.data()), thrust::raw_pointer_cast(m_chw.data()), m_hwc.size());
    cudaDeviceSynchronize();
}

float FlipMetricImpl::gaussian(const float x, const float y, const float sigma) {
    return expf(-(x * x + y * y) / (2.0f * sigma * sigma));
}

void FlipMetricImpl::sRGB2YCxCz(const unsigned char* input, ColorVec& output) {
    __cuFlip_sRGB2YCxCz<<<128, 4>>>(input, thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::YCxCz2Gray(const ColorVec& input, ColorVec& output) {
    __cuFlip_YCxCz2Gray<<<128, 4>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::YCxCz2CIELab(const ColorVec& input, ColorVec& output) {
    __cuFlip_YCxCz2CIELab<<<128, 4>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::huntAdjustment(const ColorVec& input, ColorVec& output) {
    __cuFlip_huntAdjustment<<<128, 4>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::normalize(const ColorVec& input, ColorVec& output, const Color& total) {
    __cuFlip_normalize<<<128, 4>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), total, output.size());
}

void FlipMetricImpl::generateSpatialFilter(ColorVec& output, unsigned int width, float radius, float deltaX) {
    __cuFlip_generateSpatialFilter<<<128, 4>>>(thrust::raw_pointer_cast(output.data()), width, radius, deltaX);
}

void FlipMetricImpl::convolve(const std::shared_ptr<Filter>& filter, const ColorVec& input, ColorVec& output) {
    __cuFlip_hwc2chw<<<128, 4>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(m_chwInput.data()), input.size());
    cudaDeviceSynchronize();

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnConvolutionForward(m_cudnnHandle,
                            &alpha, m_inputDescriptor, thrust::raw_pointer_cast(m_chwInput.data()),
                            filter->getFilterDescriptor(), thrust::raw_pointer_cast(filter->getDataCHW().data()),
                            filter->getConvolutionDescriptor(), filter->getConvolutionAlgorithm(),
                            filter->getWorkspace(), filter->getWorkspaceSize(),
                            &beta, m_outputDescriptor, thrust::raw_pointer_cast(m_chwOutput.data()));
    __cuFlip_chw2hwc<<<128, 4>>>(thrust::raw_pointer_cast(m_chwOutput.data()), thrust::raw_pointer_cast(output.data()), output.size());
    cudaDeviceSynchronize();
}

void FlipMetricImpl::computeColorDifference(const ColorVec& reference, const ColorVec& test, ColorVec& output) {
    __cuFlip_computeColorDifference<<<128, 4>>>(thrust::raw_pointer_cast(reference.data()), thrust::raw_pointer_cast(test.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::computeFeatureDifference(const ColorVec& edgesReference, const ColorVec& edgesTest, const ColorVec& pointsReference, const ColorVec& pointsTest, ColorVec& output) {
    __cuFlip_computeFeatureDifference<<<128, 4>>>(thrust::raw_pointer_cast(edgesReference.data()), thrust::raw_pointer_cast(edgesTest.data()), thrust::raw_pointer_cast(pointsReference.data()), thrust::raw_pointer_cast(pointsTest.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::computeFlipError(const ColorVec& colorDiff, const ColorVec& featureDiff, thrust::device_vector<float>& output) {
    __cuFlip_computeFlipError<<<128, 4>>>(thrust::raw_pointer_cast(colorDiff.data()), thrust::raw_pointer_cast(featureDiff.data()), thrust::raw_pointer_cast(output.data()), output.size());
}

void FlipMetricImpl::createColorFilter() {
    const float deltaX = 1.0f / m_ppd;
    const float pi_sq = float(M_PI * M_PI);
    // constants for Gaussians -- see paper for details.
    Color b1 = {0.0047f, 0.0053f, 0.04f };
    Color b2 = { 1.0e-5f, 1.0e-5f, 0.025f };

    float maxScaleParameter = std::max(std::max(std::max(b1.x, b1.y), std::max(b1.z, b2.x)), std::max(b2.y, b2.z));
    int radius = int(std::ceil(3.0f * sqrtf(maxScaleParameter / (2.0f * pi_sq)) * m_ppd));

    unsigned int width = 2 * radius + 1;
    m_colorFilter = std::make_shared<Filter>(m_cudnnHandle, m_inputDescriptor, m_outputDescriptor, width, width);

    ColorVec& filterData = m_colorFilter->getDataHWC();

    generateSpatialFilter(filterData, width, radius, deltaX);
    cudaDeviceSynchronize();

    Color totalFilterColor = thrust::reduce(filterData.begin(), filterData.end(), Color{0.0f, 0.0f, 0.0f});
    normalize(filterData, filterData, totalFilterColor);
    cudaDeviceSynchronize();

    m_colorFilter->toCHW();
}

void FlipMetricImpl::createDetectionFilter(ColorVec& output, float stdDev, float radius, int width, bool pointDetector) {
    float weightX, weightY;
    float negativeWeightsSumX = 0.0f;
    float positiveWeightsSumX = 0.0f;
    float negativeWeightsSumY = 0.0f;
    float positiveWeightsSumY = 0.0f;

    for (int y = 0; y < width; y++) {
        int yy = y - radius;
        for (int x = 0; x < width; x++) {
            int xx = x - radius;
            float G = gaussian(float(xx), float(yy), stdDev);
            if (pointDetector) {
                weightX = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * G;
                weightY = (float(yy) * float(yy) / (stdDev * stdDev) - 1.0f) * G;
            } else {
                weightX = -float(xx) * G;
                weightY = -float(yy) * G;
            }

            output[y * width + x] = Color{weightX, weightY, 0.0f};

            if (weightX > 0.0f) {
                positiveWeightsSumX += weightX;
            } else {
                negativeWeightsSumX += -weightX;
            }

            if (weightY > 0.0f) {
                positiveWeightsSumY += weightY;
            } else {
                negativeWeightsSumY += -weightY;
            }
        }
    }

    // Normalize positive weights to sum to 1 and negative weights to sum to -1
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
            Color p = output[y * width + x];
            output[y * width + x] = Color{p.x / (p.x > 0.0f ? positiveWeightsSumX : negativeWeightsSumX), p.y / (p.y > 0.0f ? positiveWeightsSumY : negativeWeightsSumY), 0.0f};
        }
    }
}

void FlipMetricImpl::createDetectionFilters() {
    const float gw = 0.082f;
    const float stdDev = 0.5f * gw * m_ppd;
    const int radius = int(std::ceil(3.0f * stdDev));

    unsigned int width = 2 * radius + 1;

    m_edgesFilter = std::make_shared<Filter>(m_cudnnHandle, m_inputDescriptor, m_outputDescriptor, width, width);
    m_pointsFilter = std::make_shared<Filter>(m_cudnnHandle, m_inputDescriptor, m_outputDescriptor, width, width);

    createDetectionFilter(m_edgesFilter->getDataHWC(), stdDev, radius, width, false);
    m_edgesFilter->toCHW();

    createDetectionFilter(m_pointsFilter->getDataHWC(), stdDev, radius, width, true);
    m_pointsFilter->toCHW();
}

void FlipMetricImpl::preprocess(ColorVec& image, ColorVec& imageGray) {
    YCxCz2Gray(image, imageGray);
    cudaDeviceSynchronize();

    convolve(m_colorFilter, image, image);
    cudaDeviceSynchronize();

    YCxCz2CIELab(image, image);
    cudaDeviceSynchronize();

    huntAdjustment(image, image);
    cudaDeviceSynchronize();
}

FlipMetricImpl::FlipMetricImpl(const unsigned char* image, unsigned int width, unsigned int height, float ppd) {
    m_ppd = ppd;
    m_imageWidth = width;
    m_imageHeight = height;

    cudnnCreate(&m_cudnnHandle);

    cudnnCreateTensorDescriptor(&m_inputDescriptor);
    cudnnSetTensor4dDescriptor(m_inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, m_imageHeight, m_imageWidth);

    cudnnCreateTensorDescriptor(&m_outputDescriptor);
    cudnnSetTensor4dDescriptor(m_outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, m_imageHeight, m_imageWidth);

    createColorFilter();
    createDetectionFilters();

    m_referencePixels.resize(m_imageWidth * m_imageHeight);
    m_referenceGrayPixels.resize(m_imageWidth * m_imageHeight);
    m_testPixels.resize(m_imageWidth * m_imageHeight);
    m_testGrayPixels.resize(m_imageWidth * m_imageHeight);
    m_colorDifference.resize(m_imageWidth * m_imageHeight);
    m_featureDifference.resize(m_imageWidth * m_imageHeight);

    m_edgesReference.resize(m_imageWidth * m_imageHeight);
    m_edgesTest.resize(m_imageWidth * m_imageHeight);
    m_pointsReference.resize(m_imageWidth * m_imageHeight);
    m_pointsTest.resize(m_imageWidth * m_imageHeight);

    m_chwInput.resize(m_imageWidth * m_imageHeight * 3);
    m_chwOutput.resize(m_imageWidth * m_imageHeight * 3);

    m_flip.resize(m_imageWidth * m_imageHeight);
    m_histogram.resize(100);
    m_histogramSeq.resize(m_histogram.size());
    thrust::sequence(m_histogramSeq.begin(), m_histogramSeq.end(), 0.0f, 1.0f / m_histogramSeq.size());

    thrust::device_vector<unsigned char> imageDevice(m_imageWidth * m_imageHeight * 3);
    thrust::copy(image, image + imageDevice.size(), imageDevice.begin());
    sRGB2YCxCz(thrust::raw_pointer_cast(imageDevice.data()), m_referencePixels);
    cudaDeviceSynchronize();

    preprocess(m_referencePixels, m_referenceGrayPixels);
}

FlipMetricImpl::~FlipMetricImpl() {
    cudnnDestroyTensorDescriptor(m_inputDescriptor);
    cudnnDestroyTensorDescriptor(m_outputDescriptor);
    cudnnDestroy(m_cudnnHandle);
}

float getWeightedPercentile(const thrust::device_vector<float> histogram, const double percent) {
    double weight;
    double weightedValue;
    double bucketStep = 1.0f / histogram.size();
    double sumWeightedDataValue = 0.0;
    for (size_t bucketId = 0; bucketId < histogram.size(); bucketId++)
    {
        weight = (bucketId + 0.5) * bucketStep;
        weightedValue = histogram[bucketId] * weight;
        sumWeightedDataValue += weightedValue;
    }

    double sum = 0;
    size_t weightedMedianIndex = 0;
    for (size_t bucketId = 0; bucketId < histogram.size(); bucketId++)
    {
        weight = (bucketId + 0.5) * bucketStep;
        weightedValue = histogram[bucketId] * weight;
        weightedMedianIndex = bucketId;
        if (sum + weightedValue > percent * sumWeightedDataValue)
            break;
        sum += weightedValue;
    }

    weight = (weightedMedianIndex + 0.5) * bucketStep;
    weightedValue = histogram[weightedMedianIndex] * weight;
    double discrepancy = percent * sumWeightedDataValue - sum;
    double linearWeight = discrepancy / weightedValue; // in [0,1]
    double percentile = (weightedMedianIndex + linearWeight) * bucketStep;
    return percentile;
}

float FlipMetricImpl::compareDevice(const unsigned char* image) {
    sRGB2YCxCz(image, m_testPixels);
    cudaDeviceSynchronize();

    preprocess(m_testPixels, m_testGrayPixels);

    computeColorDifference(m_referencePixels, m_testPixels, m_colorDifference);
    convolve(m_edgesFilter, m_referenceGrayPixels, m_edgesReference);
    convolve(m_edgesFilter, m_testGrayPixels, m_edgesTest);
    convolve(m_pointsFilter, m_referenceGrayPixels, m_pointsReference);
    convolve(m_pointsFilter, m_testGrayPixels, m_pointsTest);
    cudaDeviceSynchronize();

    computeFeatureDifference(m_edgesReference, m_edgesTest, m_pointsReference, m_pointsTest, m_featureDifference);
    cudaDeviceSynchronize();

    computeFlipError(m_colorDifference, m_featureDifference, m_flip);
    cudaDeviceSynchronize();

    thrust::sort(m_flip.begin(), m_flip.end());
    thrust::upper_bound(m_flip.begin(), m_flip.end(), m_histogramSeq.begin(), m_histogramSeq.end(), m_histogram.begin());
    thrust::adjacent_difference(m_histogram.begin(), m_histogram.end(), m_histogram.begin());

    return getWeightedPercentile(m_histogram, 0.5f);
}

float FlipMetricImpl::compareHost(const unsigned char* image) {
    thrust::device_vector<unsigned char> imageDevice(m_imageWidth * m_imageHeight * 3);
    thrust::copy(image, image + imageDevice.size(), imageDevice.begin());
    return compareDevice(thrust::raw_pointer_cast(imageDevice.data()));
}
