#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <node.hpp>
#include <vector>

class DecisionTree
{
public:
    DecisionTree(int samplesLimt = 2, int maxDepth = 2);
    virtual ~DecisionTree();

    void build(const std::vector<std::vector<float>>& X);

private:
    Node* root;
};

#endif // DECISION_TREE_HPP
