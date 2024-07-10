#include <glass/memory.hpp>
#include <glass/simd/distance.hpp>

template <typename dist_t = float>
struct node {
  dist_t distance;
  int id;
};

template <typename dist_t = float>
struct maxPQIFCS {
  int capacity_, cnt;
  bool full;
  std::vector<node<dist_t>, glass::align_alloc<node<dist_t>>> data_;
  maxPQIFCS() : capacity_(0), cnt(0), full(false){};
  maxPQIFCS(int capacity) : capacity_(capacity), data_(capacity + 5), cnt(0), full(false) {}

  void resize(int capacity) {
    capacity_ = capacity;
    data_.resize(capacity + 5);
    cnt = 0;
    full = false;
  }

  inline std::pair<int, dist_t> top() { return std::make_pair(data_[1].id, data_[1].distance); }
  inline int size() { return cnt; }
  void push_down(int x) {
    int t;
    while ((x << 1) <= cnt) {
      t = x << 1;
      if ((t | 1) <= cnt && data_[t | 1].distance > data_[t].distance) t |= 1;
      if (data_[t].distance <= data_[x].distance) break;
      std::swap(data_[x], data_[t]);
      x = t;
    }
  }
  void maybe_pop_emplace(int ids, dist_t dists) {
    if (!full) {
      emplace(ids, dists);
      return;
    }
    if (data_[1].distance <= dists) return;
    data_[1].id = ids;
    data_[1].distance = dists;
    push_down(1);
  }
  void must_pop_emplace(int ids, dist_t dists) {
    if (data_[1].distance <= dists) return;
    data_[1].id = ids;
    data_[1].distance = dists;
    push_down(1);
  }
  void emplace(int ids, dist_t dists) {
    data_[++cnt] = {dists, ids};
    if (cnt == capacity_) full = true;
    int now = cnt;
    while (now != 1 && data_[now].distance > data_[now >> 1].distance)
      std::swap(data_[now], data_[now >> 1]), now >>= 1;
  }
  void build() {
    for (int i = cnt / 2; i; i--) push_down(i);
  }
};
