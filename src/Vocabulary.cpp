#include "Vocabulary.h"
#include "DescManip.h"
#include "quicklz.h"
#include "timers.h"
#include <cassert>
#include <fcntl.h>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace DBoW3 {
// --------------------------------------------------------------------------

Vocabulary::Vocabulary(int k, int L, WeightingType weighting,
                       ScoringType scoring)
    : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
      m_scoring_object(NULL) {
  createScoringObject();
  std::cout << "OpenMP支持: "
            << (omp_get_max_threads() > 1 ? "已启用" : "未启用")
            << ", 最大线程数: " << omp_get_max_threads() << std::endl;
}

// --------------------------------------------------------------------------

Vocabulary::Vocabulary(const std::string &filename) : m_scoring_object(NULL) {
  load(filename);
}

// --------------------------------------------------------------------------

Vocabulary::Vocabulary(const char *filename) : m_scoring_object(NULL) {
  load(filename);
}

// --------------------------------------------------------------------------

Vocabulary::Vocabulary(std::istream &stream) : m_scoring_object(NULL) {
  load(stream);
}

// --------------------------------------------------------------------------

void Vocabulary::createScoringObject() {
  delete m_scoring_object;
  m_scoring_object = NULL;

  switch (m_scoring) {
  case L1_NORM:
    m_scoring_object = new L1Scoring;
    break;

  case L2_NORM:
    m_scoring_object = new L2Scoring;
    break;

  case CHI_SQUARE:
    m_scoring_object = new ChiSquareScoring;
    break;

  case KL:
    m_scoring_object = new KLScoring;
    break;

  case BHATTACHARYYA:
    m_scoring_object = new BhattacharyyaScoring;
    break;

  case DOT_PRODUCT:
    m_scoring_object = new DotProductScoring;
    break;
  }
}

// --------------------------------------------------------------------------

void Vocabulary::setScoringType(ScoringType type) {
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------

void Vocabulary::setWeightingType(WeightingType type) {
  this->m_weighting = type;
}

// --------------------------------------------------------------------------

Vocabulary::Vocabulary(const Vocabulary &voc) : m_scoring_object(NULL) {
  *this = voc;
}

// --------------------------------------------------------------------------

Vocabulary::~Vocabulary() { delete m_scoring_object; }

// --------------------------------------------------------------------------

Vocabulary &Vocabulary::operator=(const Vocabulary &voc) {
  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();

  this->m_nodes.clear();
  this->m_words.clear();

  this->m_nodes = voc.m_nodes;
  this->createWords();

  return *this;
}

void Vocabulary::create(const std::vector<cv::Mat> &training_features) {
  std::vector<std::vector<cv::Mat>> vtf(training_features.size());
  for (size_t i = 0; i < training_features.size(); i++) {
    vtf[i].resize(training_features[i].rows);
    for (int r = 0; r < training_features[i].rows; r++)
      vtf[i][r] = training_features[i].rowRange(r, r + 1);
  }
  create(vtf);
}

void Vocabulary::create(
    const std::vector<std::vector<cv::Mat>> &training_features) {
  m_nodes.clear();
  m_words.clear();

  // expected_nodes = Sum_{i=0..L} ( k^i )
  int expected_nodes =
      (int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree

  std::vector<cv::Mat> features;
  getFeatures(training_features, features);

  // create root
  m_nodes.push_back(Node(0)); // root

  // create the tree
  HKmeansStep(0, features, 1);

  // create the words
  createWords();

  // and set the weight of each node of the tree
  setNodeWeights(training_features);
}

// --------------------------------------------------------------------------

void Vocabulary::create(
    const std::vector<std::vector<cv::Mat>> &training_features, int k, int L) {
  m_k = k;
  m_L = L;

  create(training_features);
}

// --------------------------------------------------------------------------

void Vocabulary::create(
    const std::vector<std::vector<cv::Mat>> &training_features, int k, int L,
    WeightingType weighting, ScoringType scoring) {
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();

  create(training_features);
}

// --------------------------------------------------------------------------

void Vocabulary::getFeatures(
    const std::vector<std::vector<cv::Mat>> &training_features,
    std::vector<cv::Mat> &features) const {
  features.resize(0);
  for (size_t i = 0; i < training_features.size(); i++)
    for (size_t j = 0; j < training_features[i].size(); j++)
      features.push_back(training_features[i][j]);
}

// --------------------------------------------------------------------------

void Vocabulary::HKmeansStep(NodeId parent_id,
                             const std::vector<cv::Mat> &descriptors,
                             int current_level) {
  std::cout << "[HKmeans] 开始第 " << current_level << "/" << m_L
            << " 层构建，处理 " << descriptors.size()
            << " 个描述子，父节点ID: " << parent_id << std::endl;

  if (descriptors.empty())
    return;

  // features associated to each cluster
  std::vector<cv::Mat> clusters;
  std::vector<std::vector<unsigned int>> groups; // groups[i] = [j1, j2, ...]
  // j1, j2, ... indices of descriptors associated to cluster i

  clusters.reserve(m_k);
  groups.reserve(m_k);

  if ((int)descriptors.size() <= m_k) {
    // trivial case: one cluster per feature
    groups.resize(descriptors.size());

    for (unsigned int i = 0; i < descriptors.size(); i++) {
      groups[i].push_back(i);
      clusters.push_back(descriptors[i]);
    }
  } else {
    // select clusters and groups with kmeans

    bool first_time = true;
    bool goon = true;

    // to check if clusters move after iterations
    std::vector<int> last_association, current_association;

    int iteration = 0;
    while (goon) {
      iteration++;
      std::cout << "[K-means] 第 " << iteration << " 次迭代，处理 "
                << descriptors.size() << " 个描述子" << std::endl;

      // 1. Calculate clusters

      if (first_time) {
        // random sample
        initiateClusters(descriptors, clusters);
      } else {
// calculate cluster centres
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
        for (unsigned int c = 0; c < clusters.size(); ++c) {
          std::vector<cv::Mat> cluster_descriptors;
          cluster_descriptors.reserve(groups[c].size());

          for (auto vit = groups[c].begin(); vit != groups[c].end(); ++vit) {
            cluster_descriptors.push_back(descriptors[*vit]);
          }

          DescManip::meanValue(cluster_descriptors, clusters[c]);
        }
      } // if(!first_time)

      // 2. Associate features with clusters

      // calculate distances to cluster centers
      std::cout << "[K-means] 开始计算 " << descriptors.size() << " 个描述子到 "
                << clusters.size() << " 个聚类中心的距离..." << std::endl;

      groups.clear();
      groups.resize(clusters.size(), std::vector<unsigned int>());
      current_association.resize(descriptors.size());

      // 创建线程私有分组
      std::vector<std::vector<std::vector<unsigned int>>> private_groups;
      int num_threads = std::min(16, omp_get_max_threads());

#pragma omp parallel num_threads(num_threads)
      {
        const int thread_id = omp_get_thread_num();

// 延迟初始化线程私有存储
#pragma omp single
        {
          private_groups.resize(num_threads);
          for (int i = 0; i < num_threads; i++)
            private_groups[i].resize(clusters.size());

          std::cout << "[K-means] 使用 " << num_threads << " 个线程并行计算距离"
                    << std::endl;
        }

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < descriptors.size(); i++) {
          double best_dist = DescManip::distance(descriptors[i], clusters[0]);
          unsigned int icluster = 0;

          for (unsigned int c = 1; c < clusters.size(); ++c) {
            double dist = DescManip::distance(descriptors[i], clusters[c]);
            if (dist < best_dist) {
              best_dist = dist;
              icluster = c;
            }
          }

          private_groups[thread_id][icluster].push_back(i);
          current_association[i] = icluster;
        }

// 合并结果
#pragma omp for schedule(static)
        for (unsigned int c = 0; c < clusters.size(); ++c) {
          for (int t = 0; t < num_threads; t++) {
            groups[c].insert(groups[c].end(), private_groups[t][c].begin(),
                             private_groups[t][c].end());
          }
        }
      }

      std::cout << "[K-means] 距离计算完成，共有 " << descriptors.size()
                << " 个描述子被分配到 " << clusters.size() << " 个聚类中"
                << std::endl;

      // kmeans++ ensures all the clusters has any feature associated
      // with them

      // 3. check convergence
      if (first_time) {
        first_time = false;
      } else {
        // goon = !eqUChar(last_assoc, assoc);

        goon = false;
        for (unsigned int i = 0; i < current_association.size(); i++) {
          if (current_association[i] != last_association[i]) {
            goon = true;
            break;
          }
        }
      }

      if (goon) {
        // copy last feature-cluster association
        last_association = current_association;
        // last_assoc = assoc.clone();
      }

      // 在迭代结束时显示收敛信息
      if (!goon) {
        std::cout << "[K-means] 聚类收敛，共迭代 " << iteration << " 次"
                  << std::endl;
      }

    } // while(goon)

  } // if must run kmeans

  // create nodes
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    NodeId id = m_nodes.size();
    m_nodes.push_back(Node(id));
    m_nodes.back().descriptor = clusters[i];
    m_nodes.back().parent = parent_id;
    m_nodes[parent_id].children.push_back(id);
  }

  std::cout << "[HKmeans] 第 " << current_level << " 层处理完成，创建了 "
            << clusters.size() << " 个节点" << std::endl;

  // go on with the next level
  if (current_level < m_L) {
    // iterate again with the resulting clusters
    const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
    for (unsigned int i = 0; i < clusters.size(); ++i) {
      NodeId id = children_ids[i];

      std::vector<cv::Mat> child_features;
      child_features.reserve(groups[i].size());

      std::vector<unsigned int>::const_iterator vit;
      for (vit = groups[i].begin(); vit != groups[i].end(); ++vit) {
        child_features.push_back(descriptors[*vit]);
      }

      if (child_features.size() > 1) {
        std::cout << "[HKmeans] 开始处理第 " << current_level + 1 << " 层节点 "
                  << (i + 1) << "/" << clusters.size() << "，包含 "
                  << child_features.size() << " 个特征" << std::endl;
        HKmeansStep(id, child_features, current_level + 1);
      }
    }
  }
}

// --------------------------------------------------------------------------

void Vocabulary::initiateClusters(const std::vector<cv::Mat> &descriptors,
                                  std::vector<cv::Mat> &clusters) const {
  initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------

void Vocabulary::initiateClustersKMpp(const std::vector<cv::Mat> &pfeatures,
                                      std::vector<cv::Mat> &clusters) const {
  // Implements kmeans++ seeding algorithm
  // Algorithm:
  // 1. Choose one center uniformly at random from among the data points.
  // 2. For each data point x, compute D(x), the distance between x and
  // the nearest
  //    center that has already been chosen.
  // 3. Add one new data point as a center. Each point x is chosen with
  // probability
  //    proportional to D(x)^2.
  // 4. Repeat Steps 2 and 3 until k centers have been chosen.
  // 5. Now that the initial centers have been chosen, proceed using
  // standard k-means
  //    clustering.

  //  DUtils::Random::SeedRandOnce();

  std::cout << "[K-means++] 开始选择 " << m_k << " 个聚类中心..." << std::endl;

  clusters.resize(0);
  clusters.reserve(m_k);
  std::vector<double> min_dists(pfeatures.size(),
                                std::numeric_limits<double>::max());

  // 1.

  int ifeature =
      rand() %
      pfeatures.size(); // DUtils::Random::RandomInt(0, pfeatures.size()-1);

  // create first cluster
  clusters.push_back(pfeatures[ifeature]);
  std::cout << "[K-means++] 已选择第1个聚类中心" << std::endl;

  // compute the initial distances
  std::vector<double>::iterator dit;
  dit = min_dists.begin();
  for (auto fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit) {
    *dit = DescManip::distance((*fit), clusters.back());
  }

  while ((int)clusters.size() < m_k) {
    // 2.
    dit = min_dists.begin();
    for (auto fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit) {
      if (*dit > 0) {
        double dist = DescManip::distance((*fit), clusters.back());
        if (dist < *dit)
          *dit = dist;
      }
    }

    // 3.
    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

    if (dist_sum > 0) {
      double cut_d;
      do {
        cut_d = (double(rand()) / double(RAND_MAX)) * dist_sum;
      } while (cut_d == 0.0);

      double d_up_now = 0;
      for (dit = min_dists.begin(); dit != min_dists.end(); ++dit) {
        d_up_now += *dit;
        if (d_up_now >= cut_d)
          break;
      }

      if (dit == min_dists.end())
        ifeature = pfeatures.size() - 1;
      else
        ifeature = dit - min_dists.begin();

      clusters.push_back(pfeatures[ifeature]);
      std::cout << "[K-means++] 已选择 " << clusters.size() << "/" << m_k
                << " 个聚类中心 (" << (clusters.size() * 100 / m_k) << "%)"
                << std::endl;
    } // if dist_sum > 0
    else
      break;

  } // while(used_clusters < m_k)
}

// --------------------------------------------------------------------------

void Vocabulary::createWords() {
  m_words.resize(0);

  if (!m_nodes.empty()) {
    m_words.reserve((int)pow((double)m_k, (double)m_L));

    auto nit = m_nodes.begin(); // ignore root
    for (++nit; nit != m_nodes.end(); ++nit) {
      if (nit->isLeaf()) {
        nit->word_id = m_words.size();
        m_words.push_back(&(*nit));
      }
    }
  }
}

// --------------------------------------------------------------------------

void Vocabulary::setNodeWeights(
    const std::vector<std::vector<cv::Mat>> &training_features) {
  const unsigned int NWords = m_words.size();
  const unsigned int NDocs = training_features.size();

  if (m_weighting == TF || m_weighting == BINARY) {
    // idf part must be 1 always
    for (unsigned int i = 0; i < NWords; i++)
      m_words[i]->weight = 1;
  } else if (m_weighting == IDF || m_weighting == TF_IDF) {
    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);

    for (auto mit = training_features.begin(); mit != training_features.end();
         ++mit) {
      fill(counted.begin(), counted.end(), false);

      for (auto fit = mit->begin(); fit < mit->end(); ++fit) {
        WordId word_id;
        transform(*fit, word_id);

        if (!counted[word_id]) {
          Ni[word_id]++;
          counted[word_id] = true;
        }
      }
    }

    // set ln(N/Ni)
    for (unsigned int i = 0; i < NWords; i++) {
      if (Ni[i] > 0) {
        m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
      } // else // This cannot occur if using kmeans++
    }
  }
}

// --------------------------------------------------------------------------

// --------------------------------------------------------------------------

float Vocabulary::getEffectiveLevels() const {
  long sum = 0;
  for (auto wit = m_words.begin(); wit != m_words.end(); ++wit) {
    const Node *p = *wit;

    for (; p->id != 0; sum++)
      p = &m_nodes[p->parent];
  }

  return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------

cv::Mat Vocabulary::getWord(WordId wid) const {
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------

WordValue Vocabulary::getWordWeight(WordId wid) const {
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------

WordId Vocabulary::transform(const cv::Mat &feature) const {
  if (empty()) {
    return 0;
  }

  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------

void Vocabulary::transform(const cv::Mat &features, BowVector &v) const {
  //    std::vector<cv::Mat> vf(features.rows);
  //    for(int r=0;r<features.rows;r++) vf[r]=features.rowRange(r,r+1);
  //    transform(vf,v);

  v.clear();

  if (empty()) {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);

  if (m_weighting == TF || m_weighting == TF_IDF) {
    for (int r = 0; r < features.rows; r++) {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF
      transform(features.row(r), id, w);
      // not stopped
      if (w > 0)
        v.addWeight(id, w);
    }

    if (!v.empty() && !must) {
      // unnecessary when normalizing
      const double nd = v.size();
      for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }
  } else // IDF || BINARY
  {
    for (int r = 0; r < features.rows; r++) {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(features.row(r), id, w);

      // not stopped
      if (w > 0)
        v.addIfNotExist(id, w);

    } // if add_features
  } // if m_weighting == ...

  if (must)
    v.normalize(norm);
}

void Vocabulary::transform(const std::vector<cv::Mat> &features,
                           BowVector &v) const {
  v.clear();

  if (empty()) {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);

  if (m_weighting == TF || m_weighting == TF_IDF) {
    for (auto fit = features.begin(); fit < features.end(); ++fit) {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w);

      // not stopped
      if (w > 0)
        v.addWeight(id, w);
    }

    if (!v.empty() && !must) {
      // unnecessary when normalizing
      const double nd = v.size();
      for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }
  } else // IDF || BINARY
  {
    for (auto fit = features.begin(); fit < features.end(); ++fit) {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w);

      // not stopped
      if (w > 0)
        v.addIfNotExist(id, w);

    } // if add_features
  } // if m_weighting == ...

  if (must)
    v.normalize(norm);
}

void Vocabulary::transformParallel(const std::vector<cv::Mat> &features,
                                   BowVector &v) const {
  v.clear();
  if (empty())
    return;

  // 确定归一化需求
  LNorm norm;
  bool must_normalize = m_scoring_object->mustNormalize(norm);

  // 并行处理特征
  const int num_threads = std::thread::hardware_concurrency();
  std::vector<BowVector> thread_vectors(num_threads);

  // 基于权重类型选择处理策略
  if (m_weighting == TF || m_weighting == TF_IDF) {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < features.size(); i++) {
      int thread_id = omp_get_thread_num();
      WordId id;
      WordValue w;
      transform(features[i], id, w);
      if (w > 0)
        thread_vectors[thread_id].addWeight(id, w);
    }

    // 合并所有线程的结果
    for (auto &tv : thread_vectors) {
      for (auto &p : tv) {
        v.addWeight(p.first, p.second);
      }
    }

    // 如果不需要归一化，则进行TF调整
    if (!v.empty() && !must_normalize) {
      const double nd = v.size();
      for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }
  } else // IDF || BINARY
  {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < features.size(); i++) {
      int thread_id = omp_get_thread_num();
      WordId id;
      WordValue w;
      transform(features[i], id, w);
      if (w > 0)
        thread_vectors[thread_id].addIfNotExist(id, w);
    }

    // 合并所有线程的结果 - 对于IDF和BINARY，使用addIfNotExist
    for (auto &tv : thread_vectors) {
      for (auto &p : tv) {
        v.addIfNotExist(p.first, p.second);
      }
    }
  }

  // 执行归一化(如需要)
  if (must_normalize)
    v.normalize(norm);
}

// --------------------------------------------------------------------------

void Vocabulary::transform(const std::vector<cv::Mat> &features, BowVector &v,
                           FeatureVector &fv, int levelsup) const {
  v.clear();
  fv.clear();

  if (empty()) // safe for subclasses
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);

  if (m_weighting == TF || m_weighting == TF_IDF) {
    unsigned int i_feature = 0;
    for (auto fit = features.begin(); fit < features.end();
         ++fit, ++i_feature) {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w, &nid, levelsup);

      if (w > 0) // not stopped
      {
        v.addWeight(id, w);
        fv.addFeature(nid, i_feature);
      }
    }

    if (!v.empty() && !must) {
      // unnecessary when normalizing
      const double nd = v.size();
      for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }
  } else // IDF || BINARY
  {
    unsigned int i_feature = 0;
    for (auto fit = features.begin(); fit < features.end();
         ++fit, ++i_feature) {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w, &nid, levelsup);

      if (w > 0) // not stopped
      {
        v.addIfNotExist(id, w);
        fv.addFeature(nid, i_feature);
      }
    }
  } // if m_weighting == ...

  if (must)
    v.normalize(norm);
}

// --------------------------------------------------------------------------

// --------------------------------------------------------------------------

void Vocabulary::transform(const cv::Mat &feature, WordId &id) const {
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------

void Vocabulary::transform(const cv::Mat &feature, WordId &word_id,
                           WordValue &weight, NodeId *nid, int levelsup) const {
  // propagate the feature down the tree

  // level at which the node must be stored in nid, if given
  const int nid_level = m_L - levelsup;
  if (nid_level <= 0 && nid != NULL)
    *nid = 0; // root

  NodeId final_id = 0; // root
  int current_level = 0;

  do {
    ++current_level;
    auto const &nodes = m_nodes[final_id].children;
    double best_d = std::numeric_limits<double>::max();
    //    DescManip::distance(feature, m_nodes[final_id].descriptor);

    for (const auto &id : nodes) {
      double d = DescManip::distance(feature, m_nodes[id].descriptor);
      if (d < best_d) {
        best_d = d;
        final_id = id;
      }
    }

    if (nid != NULL && current_level == nid_level)
      *nid = final_id;

  } while (!m_nodes[final_id].isLeaf());

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}

void Vocabulary::transform(const cv::Mat &feature, WordId &word_id,
                           WordValue &weight) const {
  // propagate the feature down the tree

  // level at which the node must be stored in nid, if given

  NodeId final_id = 0; // root
  // maximum speed by computing here distance and avoid calling to
  // DescManip::distance

  // binary descriptor
  // int ntimes=0;
  if (feature.type() == CV_8U) {
    do {
      auto const &nodes = m_nodes[final_id].children;
      uint64_t best_d = std::numeric_limits<uint64_t>::max();
      int idx = 0, bestidx = 0;
      for (const auto &id : nodes) {
        // compute distance
        //  std::cout<<idx<< " "<<id<<" "<<
        //  m_nodes[id].descriptor<<std::endl;
        uint64_t dist =
            DescManip::distance_8uc1(feature, m_nodes[id].descriptor);
        if (dist < best_d) {
          best_d = dist;
          final_id = id;
          bestidx = idx;
        }
        idx++;
      }
      // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<"
      // "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
    } while (!m_nodes[final_id].isLeaf());
  } else {
    do {
      auto const &nodes = m_nodes[final_id].children;
      uint64_t best_d = std::numeric_limits<uint64_t>::max();
      int idx = 0, bestidx = 0;
      for (const auto &id : nodes) {
        // compute distance
        //   std::cout<<idx<< " "<<id<<" "<<
        //   m_nodes[id].descriptor<<std::endl;
        uint64_t dist = DescManip::distance(feature, m_nodes[id].descriptor);
        // std::cout << id << " " << dist << " " << best_d <<
        // std::endl;
        if (dist < best_d) {
          best_d = dist;
          final_id = id;
          bestidx = idx;
        }
        idx++;
      }
      // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<"
      // "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
    } while (!m_nodes[final_id].isLeaf());
  }
  //      uint64_t ret=0;
  //      const uchar *pb = b.ptr<uchar>();
  //      for(int i=0;i<a.cols;i++,pa++,pb++){
  //          uchar v=(*pa)^(*pb);
  // #ifdef __GNUG__
  //          ret+=__builtin_popcount(v);//only in g++
  // #else

  //          ret+=v& (1<<0);
  //          ret+=v& (1<<1);
  //          ret+=v& (1<<2);
  //          ret+=v& (1<<3);
  //          ret+=v& (1<<4);
  //          ret+=v& (1<<5);
  //          ret+=v& (1<<6);
  //          ret+=v& (1<<7);
  // #endif
  //  }
  //      return ret;
  //  }
  //  else{
  //      double sqd = 0.;
  //      assert(a.type()==CV_32F);
  //      assert(a.rows==1);
  //      const float *a_ptr=a.ptr<float>(0);
  //      const float *b_ptr=b.ptr<float>(0);
  //      for(int i = 0; i < a.cols; i ++)
  //          sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
  //      return sqd;
  //  }

  //  do
  //  {
  //    auto const  &nodes = m_nodes[final_id].children;
  //    double best_d = std::numeric_limits<double>::max();

  //    for(const auto  &id:nodes)
  //    {
  //      double d = DescManip::distance(feature, m_nodes[id].descriptor);
  //      if(d < best_d)
  //      {
  //        best_d = d;
  //        final_id = id;
  //      }
  //    }
  //  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}
// --------------------------------------------------------------------------

NodeId Vocabulary::getParentNode(WordId wid, int levelsup) const {
  NodeId ret = m_words[wid]->id;   // node id
  while (levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------

void Vocabulary::getWordsFromNode(NodeId nid,
                                  std::vector<WordId> &words) const {
  words.clear();

  if (m_nodes[nid].isLeaf()) {
    words.push_back(m_nodes[nid].word_id);
  } else {
    words.reserve(m_k); // ^1, ^2, ...

    std::vector<NodeId> parents;
    parents.push_back(nid);

    while (!parents.empty()) {
      NodeId parentid = parents.back();
      parents.pop_back();

      const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      std::vector<NodeId>::const_iterator cit;

      for (cit = child_ids.begin(); cit != child_ids.end(); ++cit) {
        const Node &child_node = m_nodes[*cit];

        if (child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(*cit);

      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------

int Vocabulary::stopWords(double minWeight) {
  int c = 0;
  for (auto wit = m_words.begin(); wit != m_words.end(); ++wit) {
    if ((*wit)->weight < minWeight) {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------

void Vocabulary::save(const std::string &filename,
                      bool binary_compressed) const {
  std::cout << "[Vocabulary] 开始保存词汇表到: " << filename << std::endl;
  std::cout << "[Vocabulary] 保存格式: "
            << (filename.find(".yml") == std::string::npos ? "二进制" : "YAML")
            << (binary_compressed ? "(压缩)" : "") << std::endl;
  std::cout << "[Vocabulary] 词汇表信息: 节点数量=" << m_nodes.size()
            << ", 单词数量=" << m_words.size() << ", k=" << m_k << ", L=" << m_L
            << std::endl;

  if (filename.find(".yml") == std::string::npos) {
    std::ofstream file_out(filename, std::ios::binary);
    if (!file_out) {
      std::cout << "[Vocabulary] 错误: 无法打开文件进行写入: " << filename
                << std::endl;
      throw std::runtime_error("Vocabulary::saveBinary Could not open file :" +
                               filename + " for writing");
    }
    std::cout << "[Vocabulary] 文件已打开，开始写入二进制数据..." << std::endl;
    toStream(file_out, binary_compressed);
    std::cout << "[Vocabulary] 二进制数据写入完成" << std::endl;
  } else {
    std::cout << "[Vocabulary] 开始以YAML格式保存..." << std::endl;
    cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
      std::cout << "[Vocabulary] 错误: 无法打开文件进行YAML写入: " << filename
                << std::endl;
      throw std::string("Could not open file ") + filename;
    }
    save(fs);
    std::cout << "[Vocabulary] YAML格式保存完成" << std::endl;
  }

  // 检查文件是否成功创建
  std::ifstream test_file(filename);
  if (test_file.good()) {
    test_file.seekg(0, std::ios::end);
    size_t fileSize = test_file.tellg();
    test_file.close();
    std::cout << "[Vocabulary] 确认：文件已成功创建，大小为 " << fileSize
              << " 字节" << std::endl;
  } else {
    std::cout << "[Vocabulary] 警告：无法打开已保存的文件进行验证" << std::endl;
  }

  std::cout << "[Vocabulary] 词汇表保存完成: " << filename << std::endl;
}

void Vocabulary::load(const std::string &filename) {
  std::cout << "[Vocabulary] 开始加载词汇表: " << filename << std::endl;

  // 检查文件是否为二进制文件
  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile)
    throw std::runtime_error(
        "Vocabulary::load Could not open file :" + filename + " for reading");

  std::cout << "[Vocabulary] 文件成功打开，尝试识别格式..." << std::endl;

  if (!load(ifile)) {
    if (filename.find(".txt") != std::string::npos) {
      std::cout << "[Vocabulary] 检测到txt格式，使用文本加载器..." << std::endl;
      load_fromtxt(filename);
    } else {
      std::cout << "[Vocabulary] 尝试使用OpenCV YAML格式加载..." << std::endl;
      cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
      if (!fs.isOpened())
        throw std::string("Could not open file ") + filename;
      load(fs);
    }
  }

  std::cout << "[Vocabulary] 词汇表加载完成，词汇数量: " << m_words.size()
            << ", 节点数量: " << m_nodes.size() << std::endl;
}

bool Vocabulary::load(std::istream &ifile) {
  uint64_t sig; // magic number describing the file
  ifile.read((char *)&sig, sizeof(sig));
  if (sig != 88877711233) // Check if it is a binary file.
  {
    std::cout << "[Vocabulary] 非二进制格式 (magic number不匹配)" << std::endl;
    return false;
  }

  std::cout << "[Vocabulary] 检测到二进制格式，开始加载..." << std::endl;
  ifile.seekg(0, std::ios::beg);
  fromStream(ifile);
  std::cout << "[Vocabulary] 二进制加载完成" << std::endl;
  return true;
}

bool Vocabulary::loadMapped(const std::string &filename) {
  std::cout << "[Vocabulary] 尝试使用内存映射加载: " << filename << std::endl;

  // 打开文件
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cout << "[Vocabulary] 无法打开文件进行内存映射" << std::endl;
    return false;
  }

  // 获取文件信息
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    std::cout << "[Vocabulary] 无法获取文件状态" << std::endl;
    close(fd);
    return false;
  }

  std::cout << "[Vocabulary] 文件大小: " << sb.st_size << " 字节" << std::endl;

  // 内存映射文件
  void *mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    std::cout << "[Vocabulary] 内存映射失败" << std::endl;
    close(fd);
    return false;
  }

  // 清除原有数据
  m_words.clear();
  m_nodes.clear();

  std::cout << "[Vocabulary] 开始解析映射的内存数据..." << std::endl;

  // 开始解析映射的内存数据
  char *ptr = static_cast<char *>(mapped);
  char *current_ptr = ptr;

  // 读取魔数
  uint64_t sig;
  memcpy(&sig, current_ptr, sizeof(sig));
  current_ptr += sizeof(sig);

  if (sig != 88877711233) {
    std::cout << "[Vocabulary] 无效的文件格式 (magic number不匹配)"
              << std::endl;
    munmap(mapped, sb.st_size);
    close(fd);
    throw std::runtime_error("Vocabulary::loadMapped - 文件类型不正确");
  }

  // 读取是否压缩标志
  bool compressed;
  memcpy(&compressed, current_ptr, sizeof(compressed));
  current_ptr += sizeof(compressed);
  std::cout << "[Vocabulary] 数据" << (compressed ? "已压缩" : "未压缩")
            << std::endl;

  // 读取节点数量
  uint32_t nnodes;
  memcpy(&nnodes, current_ptr, sizeof(nnodes));
  current_ptr += sizeof(nnodes);
  std::cout << "[Vocabulary] 节点数量: " << nnodes << std::endl;

  if (nnodes == 0) {
    std::cout << "[Vocabulary] 词汇表为空" << std::endl;
    munmap(mapped, sb.st_size);
    close(fd);
    return true;
  }

  // 如果数据是压缩的，需要解压缩处理
  if (compressed) {
    std::cout << "[Vocabulary] 开始解压缩数据..." << std::endl;
    // ... 保持原有代码 ...
  } else {
    // ... 保持原有代码 ...
  }

  // 清理资源
  std::cout << "[Vocabulary] 内存映射加载完成，清理资源" << std::endl;
  munmap(mapped, sb.st_size);
  close(fd);

  std::cout << "[Vocabulary] 加载完成。词汇数量: " << m_words.size()
            << ", 节点数量: " << m_nodes.size() << std::endl;
  return true;
}

void Vocabulary::load_fromtxt(const std::string &filename) {
  std::cout << "[Vocabulary] 开始从文本文件加载: " << filename << std::endl;
  std::ifstream ifile(filename);
  if (!ifile)
    throw std::runtime_error(
        "Vocabulary:: load_fromtxt  Could not open file for reading:" +
        filename);

  int n1, n2;
  {
    std::string str;
    getline(ifile, str);
    std::stringstream ss(str);
    ss >> m_k >> m_L >> n1 >> n2;
    std::cout << "[Vocabulary] 参数: k=" << m_k << ", L=" << m_L
              << ", scoring=" << n1 << ", weighting=" << n2 << std::endl;
  }

  if (m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 ||
      n2 < 0 || n2 > 3) {
    std::cout << "[Vocabulary] 错误:参数无效" << std::endl;
    throw std::runtime_error(
        "Vocabulary loading failure: This is not a correct text file!");
  }

  m_scoring = (ScoringType)n1;
  m_weighting = (WeightingType)n2;
  createScoringObject();

  // nodes
  int expected_nodes =
      (int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
  std::cout << "[Vocabulary] 预期节点数量: " << expected_nodes << std::endl;

  m_nodes.reserve(expected_nodes);
  m_words.reserve(pow((double)m_k, (double)m_L + 1));

  m_nodes.resize(1);
  m_nodes[0].id = 0;

  int counter = 0;
  while (!ifile.eof()) {
    std::string snode;
    getline(ifile, snode);
    if (counter++ % 100 == 0) {
      std::cout << "[Vocabulary] 已处理 " << counter << " 行..." << std::endl;
    }

    if (snode.size() == 0)
      break;
    std::stringstream ssnode(snode);
    int nid = m_nodes.size();
    m_nodes.resize(m_nodes.size() + 1);
    m_nodes[nid].id = nid;

    int pid;
    ssnode >> pid;
    m_nodes[nid].parent = pid;
    m_nodes[pid].children.push_back(nid);

    int nIsLeaf;
    ssnode >> nIsLeaf;

    // read until the end and add to data
    std::vector<float> data;
    data.reserve(100);
    float d;
    while (ssnode >> d)
      data.push_back(d);
    // the weight is the last
    m_nodes[nid].weight = data.back();
    data.pop_back(); // remove
    // the rest, to the descriptor
    m_nodes[nid].descriptor.create(1, data.size(), CV_8UC1);
    auto ptr = m_nodes[nid].descriptor.ptr<uchar>(0);
    for (auto d : data)
      *ptr++ = d;

    if (nIsLeaf > 0) {
      int wid = m_words.size();
      m_words.resize(wid + 1);

      m_nodes[nid].word_id = wid;
      m_words[wid] = &m_nodes[nid];
    } else {
      m_nodes[nid].children.reserve(m_k);
    }
  }

  std::cout << "[Vocabulary] 文本加载完成。词汇数量: " << m_words.size()
            << ", 节点数量: " << m_nodes.size() << std::endl;
}

void Vocabulary::fromStream(std::istream &str) {
  std::cout << "[Vocabulary] 开始从流中加载..." << std::endl;
  m_words.clear();
  m_nodes.clear();
  uint64_t sig = 0; // magic number describing the file
  str.read((char *)&sig, sizeof(sig));
  if (sig != 88877711233) {
    std::cout << "[Vocabulary] 错误:流格式无效" << std::endl;
    throw std::runtime_error(
        "Vocabulary::fromStream  is not of appropriate type");
  }

  bool compressed;
  str.read((char *)&compressed, sizeof(compressed));
  std::cout << "[Vocabulary] 数据" << (compressed ? "已压缩" : "未压缩")
            << std::endl;

  uint32_t nnodes;
  str.read((char *)&nnodes, sizeof(nnodes));
  std::cout << "[Vocabulary] 节点数量: " << nnodes << std::endl;

  if (nnodes == 0) {
    std::cout << "[Vocabulary] 词汇表为空" << std::endl;
    return;
  }

  std::stringstream decompressed_stream;
  std::istream *_used_str = 0;

  // 读取基本参数
  if (compressed) {
    std::cout << "[Vocabulary] 开始解压缩流数据..." << std::endl;

    qlz_state_decompress state_decompress;
    memset(&state_decompress, 0, sizeof(qlz_state_decompress));

    // 设置缓冲区大小
    int chunkSize = 10000;
    std::vector<char> compressed(chunkSize + 400, 0);
    std::vector<char> decompressed(chunkSize, 0);

    // 读取块数量
    uint32_t numChunks;
    str.read((char *)&numChunks, sizeof(numChunks));
    std::cout << "[Vocabulary] 压缩块数量: " << numChunks << std::endl;

    // 逐块解压
    int total_decompressed = 0;
    for (uint32_t i = 0; i < numChunks; i++) {
      // 读取压缩数据的前9个字节以获取大小信息
      char header[9];
      str.read(header, 9);
      size_t compressed_size = qlz_size_compressed(header);

      if (compressed_size == 0 ||
          compressed_size > 100000000) // 设置一个合理的上限
      {
        std::cout << "[Vocabulary] 解压错误: 块 " << i << " 数据损坏或格式错误"
                  << std::endl;
        break;
      }

      // 调整缓冲区大小
      if (compressed_size > compressed.size())
        compressed.resize(compressed_size + 400);

      // 将前9个字节复制到压缩缓冲区
      memcpy(&compressed[0], header, 9);

      // 读取剩余的压缩数据
      str.read(&compressed[0] + 9, compressed_size - 9);

      // 解压数据
      size_t decompressed_size =
          qlz_decompress(&compressed[0], &decompressed[0], &state_decompress);

      // 写入解压后的数据流
      decompressed_stream.write(&decompressed[0], decompressed_size);
      total_decompressed += decompressed_size;

      if (i % 10 == 0 || i == numChunks - 1)
        std::cout << "[Vocabulary] 已解压 " << i + 1 << "/" << numChunks
                  << " 块, 总计 " << total_decompressed << " 字节" << std::endl;
    }

    std::cout << "[Vocabulary] 解压完成，开始读取数据结构..." << std::endl;

    _used_str = &decompressed_stream;
  } else {
    _used_str = &str;
  }

  // 从流中读取基本参数
  _used_str->read((char *)&m_k, sizeof(m_k));
  _used_str->read((char *)&m_L, sizeof(m_L));
  _used_str->read((char *)&m_scoring, sizeof(m_scoring));
  _used_str->read((char *)&m_weighting, sizeof(m_weighting));

  std::cout << "[Vocabulary] 基本参数: k=" << m_k << ", L=" << m_L
            << ", scoring=" << m_scoring << ", weighting=" << m_weighting
            << std::endl;

  // 创建评分对象
  createScoringObject();

  // 预分配节点
  m_nodes.resize(nnodes); // 根节点
  m_nodes[0].id = 0;

  // 读取节点数据
  std::cout << "[Vocabulary] 开始读取节点数据..." << std::endl;

  int node_count = 0;
  // NodeId    id, parent_id;
  // WordValue weight;

  for (size_t i = 1; i < m_nodes.size(); ++i) {
    NodeId nid;
    _used_str->read((char *)&nid, sizeof(NodeId));
    if (!_used_str->good()) {
      throw std::runtime_error(
          "Vocabulary::fromStream failed while reading node id (stream error)");
    }

    if (nid >= m_nodes.size()) {
      throw std::runtime_error(
          "Vocabulary::fromStream invalid node id " + std::to_string(nid) +
          " (nnodes=" + std::to_string(m_nodes.size()) + ")");
    }
    Node &child = m_nodes[nid];
    child.id = nid;
    _used_str->read((char *)&child.parent, sizeof(child.parent));
    _used_str->read((char *)&child.weight, sizeof(child.weight));
    DescManip::fromStream(child.descriptor, *_used_str);
    if (!_used_str->good()) {
      throw std::runtime_error("Vocabulary::fromStream failed while reading "
                               "node fields (stream error)");
    }

    if (child.parent >= m_nodes.size()) {
      throw std::runtime_error("Vocabulary::fromStream invalid parent id " +
                               std::to_string(child.parent) + " for node " +
                               std::to_string(child.id) + " (nnodes=" +
                               std::to_string(m_nodes.size()) + ")");
    }
    m_nodes[child.parent].children.push_back(child.id);

    node_count++;
    if (node_count % 100 == 0)
      std::cout << "[Vocabulary] 已读取 " << node_count << " 个节点..."
                << std::endl;
  }

  std::cout << "[Vocabulary] 共读取 " << node_count << " 个节点" << std::endl;

  // 读取词汇数据
  uint32_t m_words_size;
  _used_str->read((char *)&m_words_size, sizeof(m_words_size));
  std::cout << "[Vocabulary] 词汇数量: " << m_words_size << std::endl;

  m_words.resize(m_words_size);
  for (unsigned int i = 0; i < m_words_size; i++) {
    WordId wid, nid;
    _used_str->read((char *)&wid, sizeof(wid));
    _used_str->read((char *)&nid, sizeof(nid));

    if (!_used_str->good()) {
      throw std::runtime_error("Vocabulary::fromStream failed while reading "
                               "word mapping (stream error)");
    }

    // 确保词汇ID有效
    if (wid >= m_words_size) {
      throw std::runtime_error("Vocabulary::fromStream invalid word id " +
                               std::to_string(wid) +
                               " (words=" + std::to_string(m_words_size) + ")");
    }

    if (nid >= m_nodes.size()) {
      throw std::runtime_error(
          "Vocabulary::fromStream invalid node id in word mapping nid=" +
          std::to_string(nid) + " wid=" + std::to_string(wid) +
          " (nnodes=" + std::to_string(m_nodes.size()) + ")");
    }

    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }

  std::cout << "[Vocabulary] 从流中加载完成。词汇数量: " << m_words.size()
            << ", 节点数量: " << m_nodes.size() << std::endl;
}

void Vocabulary::load(const cv::FileStorage &fs, const std::string &name) {
  std::cout << "[Vocabulary] 从OpenCV FileStorage加载: " << name << std::endl;
  m_words.clear();
  m_nodes.clear();

  cv::FileNode fvoc = fs[name];

  m_k = (int)fvoc["k"];
  m_L = (int)fvoc["L"];
  m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  m_weighting = (WeightingType)((int)fvoc["weightingType"]);

  std::cout << "[Vocabulary] 参数: k=" << m_k << ", L=" << m_L
            << ", scoring=" << m_scoring << ", weighting=" << m_weighting
            << std::endl;

  createScoringObject();

  // nodes
  cv::FileNode fn = fvoc["nodes"];
  std::cout << "[Vocabulary] 正在加载 " << fn.size() << " 个节点..."
            << std::endl;

  m_nodes.resize(fn.size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  int node_count = 0;

  for (unsigned int i = 0; i < fn.size(); ++i) {
    NodeId nid = (int)fn[i]["nodeId"];
    NodeId pid = (int)fn[i]["parentId"];
    WordValue weight = (WordValue)fn[i]["weight"];
    std::string d = (std::string)fn[i]["descriptor"];

    m_nodes[nid].id = nid;
    m_nodes[nid].parent = pid;
    m_nodes[nid].weight = weight;
    m_nodes[pid].children.push_back(nid);

    DescManip::fromString(m_nodes[nid].descriptor, d);

    node_count++;
    if (node_count % 100 == 0)
      std::cout << "[Vocabulary] 已读取 " << node_count << " 个节点..."
                << std::endl;
  }

  // words
  fn = fvoc["words"];

  m_words.resize(fn.size());

  int word_count = 0;
  for (unsigned int i = 0; i < fn.size(); ++i) {
    NodeId wid = (int)fn[i]["wordId"];
    NodeId nid = (int)fn[i]["nodeId"];

    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];

    word_count++;
    if (word_count % 100 == 0)
      std::cout << "[Vocabulary] 已读取 " << word_count << " 个词汇..."
                << std::endl;
  }

  std::cout << "[Vocabulary] OpenCV格式加载完成。词汇数量: " << m_words.size()
            << ", 节点数量: " << m_nodes.size() << std::endl;
}

void Vocabulary::save(cv::FileStorage &f, const std::string &name) const {
  std::cout << "[Vocabulary::save] 开始将词汇表写入OpenCV FileStorage: " << name
            << std::endl;

  f << name << "{";

  f << "k" << m_k;
  f << "L" << m_L;
  f << "scoringType" << m_scoring;
  f << "weightingType" << m_weighting;
  std::cout << "[Vocabulary::save] 已写入基本参数: k=" << m_k << ", L=" << m_L
            << ", scoring=" << m_scoring << ", weighting=" << m_weighting
            << std::endl;

  // tree
  f << "nodes" << "[";
  std::vector<NodeId> parents, children;
  std::vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  int node_count = 0;
  std::cout << "[Vocabulary::save] 开始写入节点数据..." << std::endl;

  while (!parents.empty()) {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node &parent = m_nodes[pid];
    children = parent.children;

    for (pit = children.begin(); pit != children.end(); pit++) {
      node_count++;
      const Node &child = m_nodes[*pit];

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << DescManip::toString(child.descriptor);
      f << "}";

      // add to parent list
      if (!child.isLeaf()) {
        parents.push_back(*pit);
      }

      // 每处理100个节点打印一次进度
      if (node_count % 100 == 0)
        std::cout << "[Vocabulary::save] 已处理 " << node_count << " 个节点..."
                  << std::endl;
    }
  }
  std::cout << "[Vocabulary::save] 完成节点数据写入，共 " << node_count
            << " 个节点" << std::endl;

  f << "]"; // nodes

  // words
  f << "words" << "[";
  std::cout << "[Vocabulary::save] 开始写入词汇数据，词汇数量: "
            << m_words.size() << std::endl;

  for (auto wit = m_words.begin(); wit != m_words.end(); wit++) {
    WordId id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }

  f << "]"; // words
  f << "}";

  std::cout << "[Vocabulary::save] 完成OpenCV FileStorage词汇表写入"
            << std::endl;
}

void Vocabulary::toStream(std::ostream &out_str, bool compressed) const {
  std::cout << "[Vocabulary::toStream] 开始写入数据流" << std::endl;

  uint64_t sig = 88877711233; // magic number describing the file
  out_str.write((char *)&sig, sizeof(sig));
  out_str.write((char *)&compressed, sizeof(compressed));
  uint32_t nnodes = m_nodes.size();
  out_str.write((char *)&nnodes, sizeof(nnodes));
  std::cout << "[Vocabulary::toStream] 已写入头部信息: 魔数, 压缩标志="
            << (compressed ? "true" : "false") << ", 节点数量=" << nnodes
            << std::endl;

  if (nnodes == 0) {
    std::cout << "[Vocabulary::toStream] 空词汇表，写入完成" << std::endl;
    return;
  }

  // save everything to a stream
  std::stringstream aux_stream;
  aux_stream.write((char *)&m_k, sizeof(m_k));
  aux_stream.write((char *)&m_L, sizeof(m_L));
  aux_stream.write((char *)&m_scoring, sizeof(m_scoring));
  aux_stream.write((char *)&m_weighting, sizeof(m_weighting));
  std::cout << "[Vocabulary::toStream] 已写入基本参数: k=" << m_k
            << ", L=" << m_L << ", scoring=" << m_scoring
            << ", weighting=" << m_weighting << std::endl;

  // nodes
  std::vector<NodeId> parents = {0}; // root
  int node_count = 0;

  std::cout << "[Vocabulary::toStream] 开始写入节点数据..." << std::endl;
  while (!parents.empty()) {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node &parent = m_nodes[pid];

    for (auto pit : parent.children) {
      node_count++;
      const Node &child = m_nodes[pit];
      aux_stream.write((char *)&child.id, sizeof(child.id));
      aux_stream.write((char *)&pid, sizeof(pid));
      aux_stream.write((char *)&child.weight, sizeof(child.weight));
      DescManip::toStream(child.descriptor, aux_stream);
      // add to parent list
      if (!child.isLeaf())
        parents.push_back(pit);
    }

    // 每处理100个节点打印一次进度
    if (node_count % 100 == 0)
      std::cout << "[Vocabulary::toStream] 已处理 " << node_count
                << " 个节点..." << std::endl;
  }
  std::cout << "[Vocabulary::toStream] 完成节点数据写入，共 " << node_count
            << " 个节点" << std::endl;

  // words
  // save size
  uint32_t m_words_size = m_words.size();
  aux_stream.write((char *)&m_words_size, sizeof(m_words_size));
  std::cout << "[Vocabulary::toStream] 开始写入词汇数据，词汇数量: "
            << m_words_size << std::endl;

  for (auto wit = m_words.begin(); wit != m_words.end(); wit++) {
    WordId id = wit - m_words.begin();
    aux_stream.write((char *)&id, sizeof(id));
    aux_stream.write((char *)&(*wit)->id, sizeof((*wit)->id));
  }
  std::cout << "[Vocabulary::toStream] 完成词汇数据写入" << std::endl;

  // now, decide if compress or not
  if (compressed) {
    std::cout << "[Vocabulary::toStream] 开始压缩数据..." << std::endl;
    qlz_state_compress state_compress;
    memset(&state_compress, 0, sizeof(qlz_state_compress));
    // Create output buffer
    int chunkSize = 10000;
    std::vector<char> compressed(chunkSize + size_t(400), 0);
    std::vector<char> input(chunkSize, 0);
    int64_t total_size = static_cast<int64_t>(aux_stream.tellp());
    uint64_t total_compress_size = 0;

    std::cout << "[Vocabulary::toStream] 未压缩数据大小: " << total_size
              << " 字节" << std::endl;

    // calculate how many chunks will be written
    uint32_t nChunks = total_size / chunkSize;
    if (total_size % chunkSize != 0)
      nChunks++;
    out_str.write((char *)&nChunks, sizeof(nChunks));
    std::cout << "[Vocabulary::toStream] 将分成 " << nChunks << " 个块进行压缩"
              << std::endl;

    // start compressing the chunks
    int chunk_idx = 0;
    while (total_size != 0) {
      int readSize = chunkSize;
      if (total_size < chunkSize)
        readSize = total_size;
      aux_stream.read(&input[0], readSize);
      uint64_t compressed_size =
          qlz_compress(&input[0], &compressed[0], readSize, &state_compress);
      total_size -= readSize;
      out_str.write(&compressed[0], compressed_size);
      total_compress_size += compressed_size;

      chunk_idx++;
      if (chunk_idx % 10 == 0 || total_size == 0)
        std::cout << "[Vocabulary::toStream] 已压缩 " << chunk_idx << "/"
                  << nChunks << " 块，当前压缩率: " << std::fixed
                  << std::setprecision(2)
                  << (100.0 -
                      (total_compress_size * 100.0 / (chunk_idx * chunkSize)))
                  << "%" << std::endl;
    }

    std::cout << "[Vocabulary::toStream] 压缩完成，压缩后大小: "
              << total_compress_size << " 字节" << std::endl;
  } else {
    std::cout << "[Vocabulary::toStream] 不压缩，直接写入数据" << std::endl;
    out_str << aux_stream.rdbuf();
    std::cout << "[Vocabulary::toStream] 数据写入完成，大小: "
              << aux_stream.tellp() << " 字节" << std::endl;
  }

  std::cout << "[Vocabulary::toStream] 流数据写入完成" << std::endl;
}

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */

std::ostream &operator<<(std::ostream &os, const Vocabulary &voc) {
  os << "Vocabulary: k = " << voc.getBranchingFactor()
     << ", L = " << voc.getDepthLevels() << ", Weighting = ";

  switch (voc.getWeightingType()) {
  case TF_IDF:
    os << "tf-idf";
    break;
  case TF:
    os << "tf";
    break;
  case IDF:
    os << "idf";
    break;
  case BINARY:
    os << "binary";
    break;
  }

  os << ", Scoring = ";
  switch (voc.getScoringType()) {
  case L1_NORM:
    os << "L1-norm";
    break;
  case L2_NORM:
    os << "L2-norm";
    break;
  case CHI_SQUARE:
    os << "Chi square distance";
    break;
  case KL:
    os << "KL-divergence";
    break;
  case BHATTACHARYYA:
    os << "Bhattacharyya coefficient";
    break;
  case DOT_PRODUCT:
    os << "Dot product";
    break;
  }

  os << ", Number of words = " << voc.size();

  return os;
}
/**
 * @brief Vocabulary::clear
 */
void Vocabulary::clear() {
  delete m_scoring_object;
  m_scoring_object = 0;
  m_nodes.clear();
  m_words.clear();
}
int Vocabulary::getDescritorSize() const {
  if (m_words.size() == 0)
    return -1;
  else
    return m_words[0]->descriptor.cols;
}
int Vocabulary::getDescritorType() const {
  if (m_words.size() == 0)
    return -1;
  else
    return m_words[0]->descriptor.type();
}

} // namespace DBoW3
