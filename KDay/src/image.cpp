#include <image/image.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

Image::Image()
    : m_shape   {0, 0, 0},
      m_data    {}
{
}

Image::Image(int width, int height, ColorType colorType)
    : m_shape   {height, width, (int)colorType},
      m_data    {}
{
    for(int c {0}; c < m_shape(2); ++c)
    {
        m_data[c].resize(m_shape(0), m_shape(1));
        m_data[c].fill(0.0f);
    }
}

Image::Image(const char* filename, ColorType colorType)
    : m_shape   {0, 0, 0},
      m_data    {}
{
    load(filename, colorType);
}

Image::Image(Image& o)
    : m_shape   {o.m_shape},
      m_data    {}
{
    for(int c {0}; c < m_shape(2); ++c)
        m_data[c] = o.m_data[c];
}

Image::Image(Image&& o)
    : m_shape   {std::move(o.m_shape)},
      m_data    {}
{
    for(int c {0}; c < m_shape(2); ++c)
        m_data[c] = std::move(o.m_data[c]);
}

Image::~Image()
{
}

Image& Image::operator=(Image& o)
{
    m_shape = o.m_shape;
    for(int c {0}; c < m_shape(2); ++c)
        m_data[c] = o.m_data[c];
    return *this;
}

Image& Image::operator=(Image&& o)
{
    m_shape = std::move(o.m_shape);
    for(int c {0}; c < m_shape(2); ++c)
        m_data[c] = std::move(o.m_data[c]);
    return *this;
}

Eigen::MatrixXf& Image::matrix(int channel)
{
    return m_data[channel];
}

void Image::load(const char* filename, ColorType colorType)
{
    uint8_t* imgData {(uint8_t*)stbi_load(filename, &m_shape(1), &m_shape(0), &m_shape(2), (int)colorType)};
    if(imgData == nullptr)
    {
        std::cerr << "Error when loading the image: " << filename << std::endl;
        exit(1);
    }
    if(colorType != AUTO)
        m_shape(2) = (int)colorType;
    for(int c {0}; c < m_shape(2); ++c)
    {
        m_data[c].resize(m_shape(0), m_shape(1));
        for(int i {0}; i < m_shape(0); ++i)
            for(int j {0}; j < m_shape(1); ++j)
                m_data[c](i, j) = (float)imgData[m_shape(2) * (i * m_shape(1) + j) + c];
    }
    stbi_image_free(imgData);
}

void Image::write(const char* filename)
{
    uint8_t* imgData {(uint8_t*)stbi__malloc(m_shape(0) * m_shape(1) * m_shape(2))};
    for(int c {0}; c < m_shape(2); ++c)
    {
        m_data[c].resize(m_shape(0), m_shape(1));
        for(int i {0}; i < m_shape(0); ++i)
            for(int j {0}; j < m_shape(1); ++j)
                imgData[m_shape(2) * (i * m_shape(1) + j) + c] = (uint8_t)std::max(0.0f, std::min(m_data[c](i, j), 255.0f));
    }

    ImageType type {getType(filename)};
    switch(type)
    {
        case JPG:
            stbi_write_jpg(filename, m_shape(1), m_shape(0), m_shape(2), imgData, 100);
            break;
        case PNG:
            stbi_write_png(filename, m_shape(1), m_shape(0), m_shape(2), imgData, m_shape(1) * m_shape(2));
            break;
        default:
            stbi_write_bmp(filename, m_shape(1), m_shape(0), m_shape(2), imgData);
            break;
    }
    stbi_image_free(imgData);
}

int Image::width()
{
    return m_shape(1);
}

int Image::height()
{
    return m_shape(0);
}

int Image::channels()
{
    return m_shape(2);
}

Eigen::Vector3i Image::shape()
{
    return m_shape;
}

float Image::calcMean(int channel)
{
    return m_data[channel].mean();
}

void Image::calcHist(Eigen::MatrixXi& hist, int minRange, int maxRange)
{
    if(m_shape(2) == 1 || m_shape(2) == 2)
    {
        hist.resize(1, maxRange - minRange + 1);
        hist.fill(0);
        for(int i {0}; i < m_shape(0); ++i)
            for(int j {0}; j < m_shape(1); ++j)
                ++hist(0, (int)m_data[0](i, j));
    }
    else if(m_shape(2) == 3 || m_shape(2) == 4)
    {
        hist.resize(3, maxRange - minRange + 1);
        hist.fill(0);
        for(int c {0}; c < m_shape(2); ++c)
            for(int i {0}; i < m_shape(0); ++i)
                for(int j {0}; j < m_shape(1); ++j)
                    ++hist(c, (int)m_data[c](i, j));
    }
}

bool Image::boundaries(int row, int col)
{
    return (row >= 0 && row < m_shape(0)) && (col >=0 && col < m_shape(1));
}

Image Image::blend(Image& img)
{
    Image res {*this};
    for(int c {0}; c < m_shape(2); ++c)
    {
        res.m_data[c] += img.m_data[c];
        res.m_data[c] /= 2.0f;
    }
    return res;
}

Image Image::operator+(Image& img)
{
    Image res {*this};
    for(int c {0}; c < m_shape(2); ++c)
        res.m_data[c] += img.m_data[c];
    return res;
}

Image Image::squared()
{
    Image res {*this};
    for(int c {0}; c < res.m_shape(2); ++c)
        res.m_data[c] = m_data[c].array().square();
    return res;
}

Image Image::squareRooted()
{
    Image res {*this};
    for(int c {0}; c < res.m_shape(2); ++c)
        res.m_data[c] = res.m_data[c].array().sqrt();
    return res;
}

Image Image::crop(int row, int col, int xrad, int yrad)
{
    Image res {xrad * 2, yrad * 2, (ColorType)m_shape(2)};
    for(int c {0}; c < res.m_shape(2); ++c)
        for(int i {0}, oi {row - yrad}; i < res.m_shape(0); ++i, ++oi)
            for(int j {0}, oj {col - xrad}; j < res.m_shape(1); ++j, ++oj)
                if(boundaries(oi, oj))
                    res.m_data[c](i, j) = m_data[c](oi, oj);
    return res;
}

bool Image::checkInputs(int row, int col)
{
    if(row < 0 || row >= m_shape(0))
    {
        std::cerr << "Index Error: 'h' , Image heigth : "
                  << m_shape(0) << " , Given : " << row << std::endl;
        exit(1);
    }
    if(col < 0 || col >= m_shape(1))
    {
        std::cerr << "Index Error: 'w' , Image width : "
                  << m_shape(1) << " , Given : " << col << std::endl;
        exit(1);
    }
    return true;
}

ImageType Image::getType(const char* filename)
{
    std::string name {filename};
    std::string ext {name.substr(name.find(".") + 1)};
    if(ext == "png")
        return PNG;
    if(ext == "jpg" || ext == "jpeg")
        return JPG;
    return INVALID;
}

float scaleValue(float x, float a, float b, float c, float d)
{
    return c + (x - a) * ((d - c) / (b - a));
}
