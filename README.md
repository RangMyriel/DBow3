

DBoW3
=====

## 
## For an improved version of this project, please see FBOW https://github.com/rmsalinas/fbow. 




DBoW3 is an improved version of the DBow2 library, an open source C++ library for indexing and converting images into a bag-of-word representation. It implements a hierarchical tree for approximating nearest neighbours in the image feature space and creating a visual vocabulary. DBoW3 also implements an image database with inverted and direct files to index images and enabling quick queries and feature comparisons. The main differences with the previous DBow2 library are:

  * DBoW3 only requires OpenCV.  DBoW2 dependency of DLIB is been removed.
  * DBoW3 is able to use both binary and floating point descriptors out of the box. No need to reimplement any class for any descriptor.
  * DBoW3 compiles both in linux and windows.  
  * Some pieces of code have been rewritten to optimize speed. The interface of DBoW3 has been simplified.
  * Possibility of using binary files. Binary files are 4-5 times faster to load/save than yml. Also, they can be compressed.
  * Compatible with DBoW2 yml files

## 
## Citing

If you use this software in an academic work, please cite:
```@online{DBoW3, author = {Rafael Muñoz-Salinas}, 
   title = {{DBoW3} DBoW3}, 
  year = 2017, 
  url = {https://github.com/rmsalinas/DBow3}, 
  urldate = {2017-02-17} 
 } 
```


## Installation notes
 
DBoW3 requires OpenCV only.

For compiling the utils/demo_general.cpp you must compile against OpenCV 3. If you have installed the contrib_modules, use cmake option -DUSE_CONTRIB=ON to enable SURF.

## How to use

Check utils/demo_general.cpp

### Classes 

DBoW3 has two main classes: `Vocabulary` and `Database`. These implement the visual vocabulary to convert images into bag-of-words vectors and the database to index images.
See utils/demo_general.cpp for an example

### Load/Store Vocabulary

The file orbvoc.dbow3 is the ORB vocabulary in ORBSLAM2 but in binary format of DBoW3:  https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary


## fork 扩展：SuperPoint + 内存加载（SPVocabularyInMem）

本仓库在上游 DBoW3 基础上，额外加入了面向 **SuperPoint 描述子** 与 **嵌入式/离线部署** 的两个能力：

### 1) SuperPoint 描述子约定（256 维 float32）

DBoW3 上游本身已经支持浮点描述子。本 fork 进一步引入了一个固定布局的描述子存储结构，用于将词汇表节点描述子按 **SuperPoint 常见的 256 维 float** 进行序列化。

- 期望 descriptor 形状：`1 x 256`
- 期望 OpenCV 类型：`CV_32FC1`

如果你的描述子不是该形状/类型，请在写入/保存前做转换与检查（否则会导致序列化内容不正确）。

### 2) SPVocabularyInMem：词汇表 mmap 快速加载/保存

新增头文件：`src/SPVocabularyInMem.h`

提供一个自定义二进制格式（带魔数校验），用 `mmap` 方式加载，典型用途：
- Jetson/边缘设备启动加速（避免 YAML 解析/大文件 IO）
- 将词汇表作为资源文件随镜像一起部署

核心 API：

```cpp
// 保存（将 DBoW3::Vocabulary 序列化为 mmap-friendly 格式）
bool SPVocMem::to_file(const DBoW3::Vocabulary& voc, const char* filename);

// 加载（从文件 mmap 并构造一个新的 DBoW3::Vocabulary*，需要 delete）
DBoW3::Vocabulary* SPVocMem::from_file(const char* filename);
```

最小 C++ 用法示例：

```cpp
#include "DBoW3.h"
#include "SPVocabularyInMem.h"

int main() {
  // 1) 训练/加载一个 Vocabulary（这里省略训练过程）
  DBoW3::Vocabulary voc;
  // voc.load("xxx.yml.gz");  // 或你自己的训练流程

  // 2) 保存为 mmap 格式
  SPVocMem::to_file(voc, "vocab_sp.mem");

  // 3) mmap 加载
  DBoW3::Vocabulary* voc2 = SPVocMem::from_file("vocab_sp.mem");
  if(!voc2) return -1;

  // ... 使用 *voc2 做 transform/query ...

  delete voc2;
  return 0;
}
```

也可以参考测试用例：`tests/test_spvoc.cpp`。

### Python（通过 pydbow3 绑定）

如果你在本仓库中同时构建了 `pydbow3`，可在 Python 侧调用：

```python
from pydbow3 import SPVocabularyInMem

# voc 是 pydbow3.Vocabulary
SPVocabularyInMem.to_file(voc, "vocab_sp.mem")
voc2 = SPVocabularyInMem.from_file("vocab_sp.mem")
```

注意：该 `.mem` 格式与上游 DBoW3 的 `.yml/.yml.gz/.dbow3` 等格式不同，是本 fork 的“快速加载”专用格式。
 


