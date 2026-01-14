#pragma once

#include <cstddef>
#include <fcntl.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <cstdint>

#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Vocabulary.h"
#include "BowVector.h"
#pragma pack(1)

#define DBOW3_MAGIC_NUM 88877711233

#define SP_DESC_COLS 256
#define SP_DESC_ROWS 1
#define SP_DESC_TYPE CV_32FC1
struct SPDescribeMem
{
    SPDescribeMem()
        : cols(SP_DESC_COLS)
        , rows(SP_DESC_ROWS)
        , type(SP_DESC_TYPE)
    {
        memset(payload, 0, sizeof(payload));
    }
    ~SPDescribeMem() {}

    SPDescribeMem(const SPDescribeMem& other)
    {
        cols = other.cols;
        rows = other.rows;
        type = other.type;
        memcpy(payload, other.payload, sizeof(payload));
    }

    uint32_t cols = SP_DESC_COLS;                  // number of columns
    uint32_t rows = SP_DESC_ROWS;                  // number of rows
    uint32_t type = SP_DESC_TYPE;                  // type of the descriptor
    float    payload[SP_DESC_ROWS * SP_DESC_COLS]; // descriptor data

    cv::Mat toMat() const
    {
        if(0)
        {
            printf("toMat: cols:%d rows:%d type:%d\n", cols, rows, type);
            // 打印payload hex
            printf("payload hex begin...\n");
            for(int i = 0; i < rows; ++i)
            {
                for(int j = 0; j < cols; ++j)
                {
                    unsigned char* p = (unsigned char*)&payload[i * cols + j];
                    printf("%02x %02x %02x %02x ",
                           *p,
                           *(p + 1),
                           *(p + 2),
                           *(p + 3));
                }
                printf("\n");
            }
            printf("payload hex end\n");
        }
        return cv::Mat((int)rows, (int)cols, (int)type, (void*)payload);
    }

    void fromMat(const cv::Mat& m)
    {
        // printf("cols:%d rows:%d type:%d\n", m.cols, m.rows, m.type());

        auto mat_size = m.size;
        // 如果为空
        if(mat_size[0] == 0 || mat_size[1] == 0)
        {
            std::cerr << "Error: Empty matrix passed to fromMat" << std::endl;
            return;
        }

        // 检查CV_32FC1
        if(m.type() != CV_32FC1)
        {
            std::cerr << "Error: Unsupported matrix type passed to fromMat"
                      << std::endl;
            return;
        }

        if(m.isContinuous())
        {
            memcpy(payload, m.ptr(), sizeof(float) * cols * rows);
        }
        else
        {
            for(int i = 0; i < rows; ++i)
            {
                memcpy(payload + i * cols, m.ptr(i), sizeof(float) * cols);
            }
        }

        if(0)
        {
            // 打印payload hex
            printf("payload hex begin...\n");
            for(int i = 0; i < rows; ++i)
            {
                for(int j = 0; j < cols; ++j)
                {
                    unsigned char* p = (unsigned char*)&payload[i * cols + j];
                    printf("%02x %02x %02x %02x ",
                           *p,
                           *(p + 1),
                           *(p + 2),
                           *(p + 3));
                }
                printf("\n");
            }
            printf("payload hex end\n");
        }
    }
};

struct SPNodeMem
{
    SPNodeMem()
        : id(0)
        , parent(0)
        , word_id(0)
        , weight(0)
    {
        memset(&descriptor, 0, sizeof(descriptor));
    }
    ~SPNodeMem() {}
    SPNodeMem(const SPNodeMem& other)
    {
        id         = other.id;
        parent     = other.parent;
        word_id    = other.word_id;
        weight     = other.weight;
        descriptor = other.descriptor;
    }
    uint32_t id;     // node id
    uint32_t parent; // parent node id
    // uint32_t      children;   // children node id(是内存布局)
    uint32_t      word_id;    // word id
    uint32_t      weight;     // weight of the node
    SPDescribeMem descriptor; // descriptor of the node
};

struct SPWordMem
{
    SPWordMem()
        : word_id(0)
        , node_id(0)
    {
        // R1: 显式成员初始化，避免对整个对象的裸内存写入
    }
    ~SPWordMem() {}
    SPWordMem(const SPWordMem& other)
    {
        word_id = other.word_id;
        node_id = other.node_id;
    }
    uint32_t word_id; // word id
    uint32_t node_id; // node id
};

struct SPVocMem
{
    SPVocMem()
        : magic_number(DBOW3_MAGIC_NUM)
        , compressed(false)
        , nnodes(0)
        , k(0)
        , L(0)
        , scoring_type(0)
        , weighting_type(0)
    {
        memset(payload, 0, sizeof(payload));
    }
    ~SPVocMem() {}
    SPVocMem(const SPVocMem& other)
    {
        magic_number   = other.magic_number;
        compressed     = other.compressed;
        nnodes         = other.nnodes;
        k              = other.k;
        L              = other.L;
        scoring_type   = other.scoring_type;
        weighting_type = other.weighting_type;
        memcpy(payload, other.payload, sizeof(payload));
    }
    uint64_t magic_number;   // magic number
    bool     compressed;     // compressed
    uint32_t nnodes;         // number of nodes
    uint32_t k;              // branching factor
    uint32_t L;              // depth levels
    uint32_t scoring_type;   // scoring method
    uint32_t weighting_type; // weighting method
    uint8_t  payload[0];     // payload

    uint32_t   getNodesNum() const { return nnodes; }
    SPNodeMem* getNodes() const { return (SPNodeMem*)(payload); }
    uint32_t   getWordsNum() const
    {
        const uint32_t* wordsize_pointer = reinterpret_cast<const uint32_t*>(
            &payload[sizeof(SPNodeMem) * nnodes]);

        return *wordsize_pointer;
    }
    void setWordsNum(uint32_t num)
    {
        uint32_t* wordsize_pointer =
            reinterpret_cast<uint32_t*>(&payload[sizeof(SPNodeMem) * nnodes]);
        *wordsize_pointer = num;
    }
    SPWordMem* getWords() const
    {
        return (SPWordMem*)(payload + sizeof(SPNodeMem) * nnodes +
                            sizeof(uint32_t));
    }

    static DBoW3::Vocabulary* from_file(const char* filename)
    {
        std::cout << "从文件加载词汇表: " << filename << std::endl;

        int fd = ::open(filename, O_RDONLY);
        if(fd < 0)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return nullptr;
        }
        std::cout << "文件成功打开" << std::endl;

        struct stat sb;
        if(fstat(fd, &sb) == -1)
        {
            std::cerr << "Error getting file size: " << filename << std::endl;
            close(fd);
            return nullptr;
        }
        std::cout << "文件大小: " << sb.st_size << " 字节" << std::endl;

        std::cout << "映射文件到内存..." << std::endl;
        SPVocMem* mapped =
            (SPVocMem*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if(mapped == MAP_FAILED)
        {
            std::cerr << "Error mapping file: " << filename << std::endl;
            close(fd);
            return nullptr;
        }

        close(fd);
        if(mapped == nullptr)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return nullptr;
        }

        // Check magic number
        std::cout << "检查魔数..." << std::endl;
        if(mapped->magic_number != DBOW3_MAGIC_NUM)
        {
            std::cerr << "Error: Invalid magic number in file: " << filename
                      << std::endl;
            munmap(mapped, sb.st_size);
            return nullptr;
        }

        const SPVocMem* voc_mem = mapped;
        // Convert SPVocMem to Vocabulary
        std::cout << "创建词汇表对象..." << std::endl;
        DBoW3::Vocabulary* voc_out =
            new DBoW3::Vocabulary(voc_mem->k,
                                  voc_mem->L,
                                  (DBoW3::WeightingType)voc_mem->weighting_type,
                                  (DBoW3::ScoringType)voc_mem->scoring_type);

        // Load nodes
        std::cout << "加载节点信息 (共 " << voc_mem->nnodes << " 个)..."
                  << std::endl;
        const SPNodeMem*                      mem_nodes = voc_mem->getNodes();
        std::vector<DBoW3::Vocabulary::Node>& nodes     = voc_out->getNodes();

        nodes.resize(voc_mem->nnodes);
        nodes[0].id         = 0;         // Set the root node id
        nodes[0].parent     = 0;         // Set the root node parent id
        nodes[0].weight     = 0;         // Set the root node weight
        nodes[0].descriptor = cv::Mat(); // Set the root node descriptor
        nodes[0].children.clear();       // Clear the root node children
        nodes[0].word_id = 0;            // Set the root node word id

        for(uint32_t i = 1; i < voc_mem->nnodes; ++i)
        {
            if(i % (voc_mem->nnodes / 10) == 0 || i == voc_mem->nnodes - 1)
            {
                std::cout << "  节点进度: " << (i * 100 / voc_mem->nnodes)
                          << "%" << std::endl;
            }

            uint32_t current_node_id = i;

            const SPNodeMem& current_node = mem_nodes[current_node_id];
            nodes[current_node_id].id     = current_node.id; // Set the node id
            nodes[current_node_id].parent = current_node.parent;
            nodes[current_node_id].weight = current_node.weight;
            nodes[current_node_id].descriptor =
                current_node.descriptor.toMat().clone();

            // Set the parent node
            if(current_node.parent < voc_mem->nnodes)
            {
                nodes[current_node.parent].children.push_back(current_node_id);
            }
            else
            {
                // Handle the case where the parent node is out of bounds
                std::cerr << "Warning: Parent node ID out of bounds: "
                          << current_node.parent << std::endl;
            }
        }

        // Load words
        uint32_t words_num = voc_mem->getWordsNum();
        std::cout << "加载单词信息 (共 " << words_num << " 个)..." << std::endl;

        const SPWordMem*                       mem_words = voc_mem->getWords();
        std::vector<DBoW3::Vocabulary::Node*>& words     = voc_out->getWords();
        words.resize(words_num);
        for(uint32_t i = 0; i < words_num; ++i)
        {
            if(words_num > 10 &&
               (i % (words_num / 10) == 0 || i == words_num - 1))
            {
                std::cout << "  单词进度: " << (i * 100 / words_num) << "%"
                          << std::endl;
            }

            uint32_t current_word_id = i;

            const SPWordMem& current_word = mem_words[current_word_id];
            // Set the word id
            nodes[current_word.node_id].word_id = current_word.word_id;
            words[current_word_id]              = &nodes[current_word.node_id];
        }

        // unmap(mapped, sizeof(SPVocMem));
        std::cout << "解除内存映射..." << std::endl;
        if(munmap(mapped, sb.st_size) == -1)
        {
            std::cerr << "Error unmapping file: " << filename << std::endl;
            throw std::runtime_error("Error unmapping file");
        }

        std::cout << "词汇表从文件 " << filename << " 加载完成!" << std::endl;
        return voc_out;
    }

    static bool to_file(const DBoW3::Vocabulary& voc, const char* filename)
    {
        std::cout << "开始将词汇表保存到文件: " << filename << std::endl;

        const std::vector<DBoW3::Vocabulary::Node>& nodes =
            const_cast<DBoW3::Vocabulary&>(voc).getNodes();
        const std::vector<DBoW3::Vocabulary::Node*>& words =
            const_cast<DBoW3::Vocabulary&>(voc).getWords();

        // Convert Vocabulary to SPVocMem
        size_t nodes_size = nodes.size();
        size_t words_size = words.size();

        std::cout << "节点数量: " << nodes_size << ", 单词数量: " << words_size
                  << std::endl;

        size_t mem_size = sizeof(SPVocMem) + sizeof(SPNodeMem) * nodes_size +
                          sizeof(uint32_t) + sizeof(SPWordMem) * words_size;
        std::cout << "需要分配内存: " << mem_size << " 字节" << std::endl;

        std::cout << "创建文件并设置大小..." << std::endl;
        int fd = ::open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if(fd < 0)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        // ftune the file size
        if(ftruncate(fd, mem_size) == -1)
        {
            std::cerr << "Error truncating file: " << filename << std::endl;
            close(fd);
            return false;
        }

        std::cout << "映射文件到内存..." << std::endl;
        SPVocMem* mapped = (SPVocMem*)mmap(
            nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        if(mapped == MAP_FAILED)
        {
            std::cerr << "Error mapping file: " << filename << std::endl;
            close(fd);
            return false;
        }

        close(fd);

        std::cout << "写入词汇表头部信息..." << std::endl;
        mapped->magic_number   = DBOW3_MAGIC_NUM;
        mapped->compressed     = false;
        mapped->nnodes         = nodes_size;
        mapped->k              = voc.getBranchingFactor();
        mapped->L              = voc.getDepthLevels();
        mapped->scoring_type   = (uint32_t)voc.getScoringType();
        mapped->weighting_type = (uint32_t)voc.getWeightingType();
        mapped->setWordsNum(words_size);

        SPNodeMem* mem_nodes = mapped->getNodes();
        SPWordMem* mem_words = mapped->getWords();

        std::cout << "写入节点信息..." << std::endl;
        // i == 0 根节点
        mem_nodes[0].id      = 0;
        mem_nodes[0].parent  = 0;
        mem_nodes[0].weight  = 0;
        mem_nodes[0].word_id = 0;
        // 调用placement new
        new(&mem_nodes[0].descriptor) SPDescribeMem();
        mem_nodes[0].descriptor.fromMat(
            const_cast<DBoW3::Vocabulary::Node&>(nodes[0]).descriptor);
        for(size_t i = 1; i < nodes_size; ++i)
        {
            if(i % (nodes_size / 10) == 0 || i == nodes_size - 1)
            {
                std::cout << "  节点进度: " << (i * 100 / nodes_size) << "%"
                          << std::endl;
            }

            const DBoW3::Vocabulary::Node& current_node = nodes[i];
            mem_nodes[i].id                             = current_node.id;
            mem_nodes[i].parent                         = current_node.parent;
            mem_nodes[i].weight                         = current_node.weight;
            new(&mem_nodes[i].descriptor) SPDescribeMem();
            mem_nodes[i].descriptor.fromMat(
                const_cast<DBoW3::Vocabulary::Node&>(current_node).descriptor);
        }

        std::cout << "写入单词信息..." << std::endl;
        for(size_t i = 0; i < words_size; ++i)
        {
            if(words_size > 10 &&
               (i % (words_size / 10) == 0 || i == words_size - 1))
            {
                std::cout << "  单词进度: " << (i * 100 / words_size) << "%"
                          << std::endl;
            }

            const DBoW3::Vocabulary::Node* current_word = words[i];
            mem_words[i].word_id                        = current_word->word_id;
            mem_words[i].node_id                        = current_word->id;
        }

        // Unmap the file
        std::cout << "解除内存映射..." << std::endl;
        if(munmap(mapped, mem_size) == -1)
        {
            std::cerr << "Error unmapping file: " << filename << std::endl;
            return false;
        }
        std::cout << "词汇表成功保存到文件: " << filename << std::endl;
        return true;
    }
};

#pragma pack()