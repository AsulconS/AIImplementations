#include <utils.hpp>
#include <mnistParsing.hpp>

bool loadTrainingSetLabelFile(const char* path, std::vector<uint8_t>& Y)
{
    std::ifstream source;
    source.open(path, std::ios_base::binary);
    if(!source)
        return false;

    uint8_t buffer8ui;
    int32_t buffer32i;
    source.read((char*)&buffer32i, 4);
    if(flipBytes(buffer32i) != 0x00000801)
        return false;

    source.read((char*)&buffer32i, 4);
    int32_t itemsNumber {flipBytes(buffer32i)};
    while(itemsNumber--)
    {
        source.read((char*)&buffer8ui, 1);
        Y.push_back(buffer8ui);
    }

    source.close();
    return true;
}

bool loadTrainingSetImageFile(const char* path, std::vector<std::vector<uint8_t>>& X)
{
    std::ifstream source;
    source.open(path, std::ios_base::binary);
    if(!source)
        return false;

    uint8_t buffer8ui;
    int32_t buffer32i;
    source.read((char*)&buffer32i, 4);
    if(flipBytes(buffer32i) != 0x00000803)
        return false;

    source.read((char*)&buffer32i, 4);
    int32_t itemsNumber {flipBytes(buffer32i)};

    int32_t perItemSize {1};
    source.read((char*)&buffer32i, 4);
    perItemSize *= flipBytes(buffer32i);
    source.read((char*)&buffer32i, 4);
    perItemSize *= flipBytes(buffer32i);
    for(int i {0}; i < itemsNumber; ++i)
    {
        X.push_back({});
        for(int j {0}; j < perItemSize; ++j)
        {
            source.read((char*)&buffer8ui, 1);
            X[i].push_back(buffer8ui);
        }
    }

    source.close();
    return true;
}
