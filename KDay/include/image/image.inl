template <typename F, typename... KArgs>
inline void Image::convolve(int channel, F f, KArgs&&... kernels)
{
    Image temp {*this};
    for(int i {0}; i < m_shape(0); ++i)
        for(int j {0}; j < m_shape(1); ++j)
            m_data[channel](i, j) = f(temp.weighRegion(i, j, channel, std::forward<KArgs>(kernels))...);
}

template <int M, int N>
inline float Image::weighRegion(int row, int col, int channel, const Kernel<M, N>& kernel)
{
    static_assert(M & 0x1 && N & 0x1, "Kernel Dimensions must be odd");
    float accum {0.0f};
    for(int m {0}, ti {row - M / 2}; m < M; ++m, ++ti)
        for(int n {0}, tj {col - N / 2}; n < N; ++n, ++tj)
            if(boundaries(ti, tj))
                accum += (float)kernel.matrix[m][n] * m_data[channel](ti, tj);
    return accum * kernel.factor;
}
