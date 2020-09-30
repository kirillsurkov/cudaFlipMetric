all:
	nvcc main.cpp FlipMetric.cpp FlipMetricImpl.cu Image.cpp -arch=compute_35 -g -rdc=true -lpng -lcudnn
