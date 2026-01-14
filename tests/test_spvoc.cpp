#include "DBoW3.h"
#include "SPVocabularyInMem.h"

#include <iostream>
#include <opencv2/core/mat.hpp>

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <vocabulary_file>" << std::endl;
        return -1;
    }

    const char* filename = argv[1];

    // Load the vocabulary from the file
    DBoW3::Vocabulary* voc = SPVocMem::from_file(filename);
    if(!voc)
    {
        std::cerr << "Failed to load vocabulary from file: " << filename
                  << std::endl;
        return -1;
    }

    // 遍历节点
    const std::vector<DBoW3::Vocabulary::Node>& nodes =
        const_cast<DBoW3::Vocabulary&>(*voc).getNodes();

    for(size_t i = 1; i < nodes.size(); ++i)
    {
        const DBoW3::Vocabulary::Node& node = nodes[i];
        std::cout << "Node ID: " << node.id << ", Parent ID: " << node.parent
                  << ", Weight: " << node.weight
                  << ", Word ID: " << node.word_id << std::endl;
        // std::cout << "Descriptor: " << node.descriptor << std::endl;
        //  hex dump
        void*  payload      = node.descriptor.data;
        size_t payload_size = node.descriptor.rows * node.descriptor.cols *
                              node.descriptor.elemSize();
        unsigned char* p = (unsigned char*)payload;
        std::cout << "Descriptor hex: ";
        for(size_t j = 0; j < payload_size; ++j)
        {
            printf("%02x ", *p);
            p++;
        }
        std::cout << "Children: ";
        for(const auto& child : node.children) { std::cout << child << " "; }
        std::cout << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    delete voc;

    return 0;
}