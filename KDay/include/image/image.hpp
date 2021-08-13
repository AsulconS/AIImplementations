#if !defined(IMAGE_HPP)
#define IMAGE_HPP

#if defined(__CMAKE_SHARED_MODE)
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

#include <image/kernel.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <iostream>

enum ImageType { INVALID, JPG, PNG };
enum ColorType { AUTO, GRAY, GRAY_ALPHA, RGB, RGB_ALPHA };

class DLL_EXPORT Image
{
public:
    Image();
    Image(int width, int height, ColorType colorType = GRAY);
    Image(const char* filename, ColorType colorType = AUTO);
    Image(Image& o);
    Image(Image&& o);
    virtual ~Image();

    Image& operator=(Image& o);
    Image& operator=(Image&& o);

    Eigen::MatrixXf& matrix(int channel = 0);

    void load(const char* filename, ColorType colorType = AUTO);
    void write(const char* filename);

    int width();
    int height();
    int channels();
    Eigen::Vector3i shape();

    float calcMean(int channel = 0);
    void calcHist(Eigen::MatrixXi& hist, int minRange = 0, int maxRange = 255);

    template <typename F, typename... KArgs>
    void convolve(int channel, F f, KArgs&&... kernels);
    template <int M, int N>
    float weighRegion(int row, int col, int channel, const Kernel<M, N>& kernel);

    bool boundaries(int row, int col);

    Image blend(Image& img);
    Image operator+(Image& img);

    Image squared();
    Image squareRooted();

    Image crop(int row, int col, int xrad, int yrad);

private:
    bool checkInputs(int row, int col);
    ImageType getType(const char* filename);

    Eigen::Vector3i m_shape;
    Eigen::MatrixXf m_data[4];
};

float DLL_EXPORT scaleValue(float x, float a, float b, float c, float d);

#include <image/image.inl>

#endif // IMAGE_HPP
