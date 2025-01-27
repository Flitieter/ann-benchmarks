#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

#include "glass/memory.hpp"
#include "hnswlib/visited_list_pool.h"

namespace glass {

namespace searcher {
// todo:edition1
// template <typename Block = hnswlib::vl_type> struct Bitset {
////  constexpr static int block_size = sizeof(Block) * 8;
////  int nbytes;
//    hnswlib::vl_type *data;
//    hnswlib::VisitedList* vl;
//  explicit Bitset(int n,int block_size=9){
////      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
////        data((uint64_t *)alloc64B(nbytes))
//    vl = new hnswlib::VisitedList(n,block_size);
//    vl->clear();
//    data=vl->get_data();
//  }
//  ~Bitset() { free(vl); }
//  inline void set(int i) {
//    vl->set(i);
//  }
//  inline bool get(int i) {
//    return vl->get(i);
//  }
//};

// todo:edition2
template <typename Block = char>
struct Bitset {
 public:
  typedef char vl_type;
  vl_type *mass;
  unsigned int numelements;
  unsigned int blocksize;
  unsigned int log_blocksize;
  unsigned int numblock;
  vl_type *changed;

  Bitset(int numelements1, int log_blocksize = 9) {
    //        log_blocksize-=3;
    this->log_blocksize = log_blocksize;
    blocksize = 1 << (log_blocksize - 3);
    numblock = std::ceil((float)numelements1 / blocksize);
    numelements = numblock * blocksize;
    mass = new vl_type[numelements];
    numblock = (numblock >> 3) + 1;
    changed = new vl_type[numblock];
    reset();
  }

  inline void reset() {  // this reset is NOT let someplace 0
    //        curV++;
    //        if (curV == 0) {
    //            memset(mass, 0, sizeof(vl_type) * numelements);
    //            curV++;
    //        }
    clear();
  }

  inline vl_type *get_data() { return mass; }

  inline void clear() {
    memset(changed, 0, numblock);
    //        memset(mass,0,numelements);
  }

  inline void set(int place) {
    int upper_place = place >> log_blocksize;
    //        std::cout<<place<<' '<<(place>>3)<<' '<<upper_place<<std::endl;
    if (((changed[upper_place >> 3] >> (upper_place & 7)) & 1) == 0) {
      //            push_down(upper_place);
      changed[upper_place >> 3] |= (1 << (upper_place & 7));
      memset(mass + (upper_place * blocksize), 0, blocksize);
    }
    mass[place >> 3] |= (1 << (place & 7));
  }
  inline bool get(int place) {
    int upper_place = place >> log_blocksize;
    if (((changed[upper_place >> 3] >> (upper_place & 7)) & 1) == 0) return 0;
    return (mass[place >> 3] >> (place & 7)) & 1;
  }

  ~Bitset() {
    delete[] mass;
    delete[] changed;
  }
};

// todo:baseline
// template <typename Block = uint64_t> struct Bitset {
//     constexpr static int block_size = sizeof(Block) * 8;
//     int nbytes;
//     Block *data;
//     explicit Bitset(int n)
//             : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
//               data((Block *)alloc64B(nbytes)) {
//         memset(data, 0, nbytes);
//     }
//     ~Bitset() { free(data); }
//     void set(int i) {
//         data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
//     }
//     bool get(int i) {
//         return (data[i / block_size] >> (i & (block_size - 1))) & 1;
//     }
//
//     void *block_address(int i) { return data + i / block_size; }
// };

template <typename dist_t = float>
struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) { return !(lhs < rhs); }
};

template <typename dist_t>
struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++size);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return pool[size].id;
  }
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

template <typename dist_t>
struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id >= 0) {
        size--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    size++;
    return true;
  }
  dist_t max() { return pool[0].distance; }
  void clear() { size = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i);
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --size;
    return ret;
  }

  int size = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

template <typename dist_t>
struct LinearPool {
  LinearPool(int n, int capacity, int = 0) : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n), vis1(n) {}

  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  void emplace_insert(int u, dist_t dist) {
    if (dist >= data_[size_ - 1].distance) {
      return;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
  }

  int top() { return data_[cur_].id; }
  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int dist(int i) const { return data_[i].distance; }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
  Bitset<uint64_t> vis;
  Bitset<uint64_t> vis1;
};

template <typename dist_t>
struct LPool {
  LPool(int capacity) : capacity_(capacity), data_(capacity_) {}

  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  void emplace_insert(int u, dist_t dist) {
    if (dist >= data_[size_ - 1].distance) [[likely]] {
      return;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
  }

  void insert_back(int u, dist_t dist) { data_[size_++] = {u, dist}; }

  void sorted() { std::sort(data_.begin(), data_.end()); }

  int top() { return data_[cur_].id; }
  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int dist(int i) const { return data_[i].distance; }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
};

template <typename dist_t>
struct HeapPool {
  HeapPool(int n, int capacity, int topk)
      : nb(n), capacity_(capacity), candidates(capacity), retset(topk), vis(n), vis1(n) {}
  bool insert(int u, dist_t dist) {
    retset.push(u, dist);
    return candidates.push(u, dist);
  }
  int pop() { return candidates.pop_min(); }
  bool has_next() const { return candidates.size > 0; }
  int id(int i) const { return retset.pool[i].id; }
  int capacity() const { return capacity_; }
  int nb, size_ = 0, capacity_;
  MinMaxHeap<dist_t> candidates;
  MaxHeap<dist_t> retset;
  // std::priority_queue<std::pair<dist_t, int>> heap;
  Bitset<uint64_t> vis;
  Bitset<uint64_t> vis1;
};

}  // namespace searcher

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f) : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const { return distance < other.distance; }
};

struct Node {
  int id;
  float distance;

  Node() = default;
  Node(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const Node &other) const { return distance < other.distance; }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) {
      break;
    }
    if (addr[left].id == nn.id) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

}  // namespace glass
