#ifndef NODE_HPP
#define NODE_HPP

struct Node
{
    int featureIndex;
    int threshold;
    Node* left;
    Node* right;
    int infoGain;

    int value;
};

#endif // NODE_HPP
