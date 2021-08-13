#include <kdtree.hpp>
#include <image/image.hpp>

#include <cstdio>
#include <cstdint>

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

#define B1 127.5
#define B2 16256.25
#define B3 2072671.875

#define SQD(v) ((v) * (v))

bool exists(const std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

void processImage(vector<vector<double>>& points, const std::string& path, double imclass)
{
    if(!exists(path))
        return;

    Image inputImg {path.c_str(), RGB};

    double n, mean;
    double w, c, v;
    w = c = v = 0.0;

    n = (double)inputImg.width() * (double)inputImg.height();
    mean = ((double)inputImg.calcMean(0) + (double)inputImg.calcMean(1) + (double)inputImg.calcMean(2)) / 3.0;

    Eigen::MatrixXi hist;
    inputImg.calcHist(hist);

    double val;
    for(int i {0}; i < 256; ++i)
    {
        val = hist(0, i) + hist(1, i) + hist(2, i);
        w += val * (((i - B1) * (i - B1) * (i - B1)) / B3);
        c += val * (1 - ((i - B1) * (i - B1)) / B2);
        v += SQD(val - mean);
    }
    w /= n;
    c /= n;
    v /= n;
    points.push_back({w, c, v, imclass});
}

void restartClass(std::map<double, pair<std::string, int>>& classmap)
{
    for(auto& it : classmap)
        it.second.second = 0;
}

double getGreaterIndex(const std::map<double, pair<std::string, int>>& classmap)
{
    double greaterIndex {0.0};
    for(const auto& it : classmap)
        if(it.second.second > classmap.at(greaterIndex).second)
            greaterIndex = it.first;
    return greaterIndex;
}

int main(int argc, char** argv)
{
    if(argc < 3)
        return 0;

    // Second Arg is k
    int k {std::stoi(argv[2])};

    // Setup classmap
    std::map<double, pair<std::string, int>> classmap;
    classmap[0.0].first = "Interval 0: 19:00->04:00";
    classmap[1.0].first = "Interval 1: 04:00->05:30";
    classmap[2.0].first = "Interval 2: 05:30->07:00";
    classmap[3.0].first = "Interval 3: 07:00->10:00";
    classmap[4.0].first = "Interval 4: 10:00->14:00";
    classmap[5.0].first = "Interval 5: 14:00->17:00";
    classmap[6.0].first = "Interval 6: 17:00->18:00";
    classmap[7.0].first = "Interval 7: 18:00->19:00";

    // Get Training Points
    std::cout << "Processing... " << std::endl;
    vector<vector<double>> trainPoints;
    for(const auto& folder : std::filesystem::directory_iterator("dataset/day_train"))
    {
        for(const auto& entry : std::filesystem::directory_iterator(folder))
        {
            std::string fpath {folder.path().string()};
            double imclass {std::stod(fpath.substr(fpath.find_last_of("/\\") + 1))};
            processImage(trainPoints, entry.path().string(), imclass);
            std::cout << entry.path().string() << std::endl;
        }
    }
    std::cout << trainPoints.size() << " images processed" << std::endl;

    // Build KD Tree
    std::cout << "Training..." << std::endl;
    KDTree kdtree(trainPoints, trainPoints[0].size() - 1);

    // Test for KD Tree
    std::cout << "Testing... " << std::endl;
    double successes {0.0};
    vector<vector<double>> testingPoints;
    for(const auto& folder : std::filesystem::directory_iterator("dataset/day_test"))
    {
        for(const auto& entry : std::filesystem::directory_iterator(folder))
        {
            std::string fpath {folder.path().string()};
            double imclass {std::stod(fpath.substr(fpath.find_last_of("/\\") + 1))};

            WPointQueue closestPoints;
            processImage(testingPoints, entry.path().string(), imclass);
            kdtree.knn(testingPoints.back(), closestPoints, k);

            restartClass(classmap);
            int limit {closestPoints.size()};
            for(int i {0}; i < limit; ++i)
            {
                ++classmap[closestPoints.top().first.back()].second;
                closestPoints.pop_top();
            }

            double greaterIndex {getGreaterIndex(classmap)};
            if(greaterIndex == imclass)
                successes += 1.0;

            std::cout << entry.path().string() << "| Predicted: " << greaterIndex << std::endl;
        }
    }
    std::cout << testingPoints.size() << " images tested" << std::endl;
    double accuracy {successes / (double)testingPoints.size()};
    std::cout << "Accuracy: " << (accuracy * 100.0) << '%' << std::endl;

    WPointQueue closestPoints;
    vector<vector<double>> predictions;
    processImage(predictions, std::string{"dataset/day_pred/"} + argv[1], -1.0);

    std::cout << "Predict for: " << std::endl;
    std::cout << "\t? | ["
              << predictions[0][0] << ',' << ' '
              << predictions[0][1] << ']' << std::endl;
    kdtree.knn(predictions[0], closestPoints, k);

    printf("\nK-Nearest Neighbours (k=%d):\n", k);
    restartClass(classmap);
    int limit {closestPoints.size()};
    for(int i {0}; i < limit; ++i)
    {
        ++classmap[closestPoints.top().first.back()].second;
        std::cout << '\t' << closestPoints.top().first.back() << " | " << '['
                  << closestPoints.top().first[0] << ',' << ' '
                  << closestPoints.top().first[1] << ']' << std::endl;
        closestPoints.pop_top();
    }
    std::cout << std::endl;

    double greaterIndex {getGreaterIndex(classmap)};

    std::cout << "Prediction: " << std::endl;
    std::cout << "\t" << greaterIndex << " | ["
              << predictions[0][0] << ',' << ' '
              << predictions[0][1] << ']' << std::endl;
    std::cout << "\t" << classmap[greaterIndex].first << std::endl;

    return 0;
}
