template <typename T>
inline T& flipBytes(T& var)
{
    uint8_t temp {};
    uint8_t* start {(uint8_t*)&var};
    uint8_t* end {start + sizeof(T) - 1};
    while(start < end)
    {
        temp = *start;
        *start = *end;
        *end = temp;
        ++start; --end;
    }
    return var;
}
