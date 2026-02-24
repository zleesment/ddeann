#ifndef DEANN_DYNAMIC_NAIVE_KDE
#define DEANN_DYNAMIC_NAIVE_KDE

#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include "KdeEstimator.hpp"
#include "Array.hpp"  

namespace deann {

template <typename T>
void dynKdeEuclideanMatmul(int_t n, int_t m, int_t d, T h, const T *__restrict X,
                          const T *__restrict Q, T *__restrict mu,
                          const T *__restrict XSqNorms, T *__restrict scratch,
                          Kernel kernel)
{
    switch (kernel)
    {
    case Kernel::EXPONENTIAL:
    case Kernel::GAUSSIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }

    if (m < 1)
      return;

    array::euclideanSqDistances(n, m, d, X, Q, XSqNorms, scratch, scratch + n * m);

    if (kernel == Kernel::EXPONENTIAL)
    {
      array::sqrt(m * n, scratch);
      array::mul(m * n, scratch, -static_cast<T>(1) / h);
    }
    else if (kernel == Kernel::GAUSSIAN)
    {
      array::mul(m * n, scratch, -static_cast<T>(1) / h / h / 2);
    }
    array::exp(m * n, scratch);
    array::rowwiseMean(m, n, scratch, mu);
}


template <typename T>
T dynKdeEuclideanMatmul(int_t n, int_t d, T h, const T *__restrict X,
                       const T *__restrict q, const T *__restrict XSqNorms,
                       T *__restrict scratch, Kernel kernel)
{
    switch (kernel)
    {
    case Kernel::EXPONENTIAL:
    case Kernel::GAUSSIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }

    array::euclideanSqDistances(n, d, X, q, XSqNorms, scratch);
    if (kernel == Kernel::EXPONENTIAL)
    {
      array::sqrt(n, scratch);
      array::mul(n, scratch, -static_cast<T>(1) / h);
    }
    else if (kernel == Kernel::GAUSSIAN)
    {
      array::mul(n, scratch, -static_cast<T>(1) / h / h / 2);
    }
    array::exp(n, scratch);
    return array::mean(n, scratch);
}

template <typename T>
void dynKdeTaxicab(int_t n, int_t m, int_t d, T h, const T *__restrict X,
                  const T *__restrict Q, T *__restrict mu,
                  T *__restrict scratch,
                  Kernel kernel)
{
    switch (kernel)
    {
    case Kernel::LAPLACIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }

    if (m < 1)
      return;

    T *__restrict taxiScratch = scratch + m * n;
    for (int_t i = 0; i < m; ++i)
    {
      const T *__restrict q = Q + i * d;
      for (int_t j = 0; j < n; ++j)
      {
        const T *__restrict x = X + j * d;
        scratch[i * n + j] = array::taxicabDistance(d, q, x, taxiScratch);
      }
    }
    array::mul(m * n, scratch, -static_cast<T>(1) / h);
    array::exp(m * n, scratch);
    array::rowwiseMean(m, n, scratch, mu);
}   

template<typename T>
class DynamicNaiveKde {
public:
  DynamicNaiveKde(double bandwidth, Kernel kernel)
    : h(bandwidth), K(kernel) {}

  void fit(int_t n_, int_t d_, const T* X) {
    d = d_;
    Xbuf.assign(X, X + n_ * d);
    rebuildIds(n_);
    rebuildNormsIfNeeded();
  }

  // returns ids of inserted points
  std::vector<int_t> insert(int_t m, const T* Xnew) {
    if (d == 0) throw std::invalid_argument("fit must be called before insert");
    std::vector<int_t> ids;
    ids.reserve(m);

    int_t oldN = n();
    Xbuf.insert(Xbuf.end(), Xnew, Xnew + m * d);

    for (int_t i = 0; i < m; i++) {
      int_t id = nextId++;
      int_t slot = oldN + i;
      idToSlot[id] = slot;
      slotToId.push_back(id);
      ids.push_back(id);
    }

    updateNormsForAppends(oldN, m);
    return ids;
  }

  void eraseById(int_t k, const int_t* ids) {
    for (int_t i = 0; i < k; i++) {
      auto it = idToSlot.find(ids[i]);
      if (it == idToSlot.end()) continue; // or throw
      int_t slot = it->second;
      int_t last = n() - 1;

      if (slot != last) {
        // swap data
        for (int_t j = 0; j < d; j++)
          Xbuf[slot*d + j] = Xbuf[last*d + j];

        // swap id bookkeeping
        int_t movedId = slotToId[last];
        slotToId[slot] = movedId;
        idToSlot[movedId] = slot;

        // norms swap if used
        if (useNorms()) XSqNorm[slot] = XSqNorm[last];
      }

      // pop last
      Xbuf.resize((last) * d);
      idToSlot.erase(it);
      slotToId.pop_back();
      if (useNorms()) XSqNorm.pop_back();
    }
  }

  void query(int_t m, const T* Q, T* Z) const {
    int_t n_ = n();
    // exact computation over all points:
    if (K == Kernel::EXPONENTIAL || K == Kernel::GAUSSIAN) {
      std::vector<T> scratch(m*n_ + m + std::max(n_, m));
      dynKdeEuclideanMatmul(
            n_, m, d, static_cast<T>(h),
            Xbuf.data(), Q, Z,
            XSqNorm.data(), scratch.data(), K);
    } else if (K == Kernel::LAPLACIAN) {
      std::vector<T> scratch(m*n_ + d);
      dynKdeTaxicab(
            n_, m, d, static_cast<T>(h),
            Xbuf.data(), Q, Z,
            scratch.data(), K);
    }
  }

  int_t n() const { return (d == 0) ? 0 : (int_t)(Xbuf.size() / d); }
  int_t dim() const { return d; }

private:
  bool useNorms() const { return (K == Kernel::EXPONENTIAL || K == Kernel::GAUSSIAN); }

  void rebuildIds(int_t n_) {
    idToSlot.clear();
    slotToId.clear();
    slotToId.reserve(n_);
    for (int_t i = 0; i < n_; i++) {
      int_t id = nextId++;
      idToSlot[id] = i;
      slotToId.push_back(id);
    }
  }

  void rebuildNormsIfNeeded() {
    if (!useNorms()) return;
    XSqNorm.resize(n());
    array::sqNorm(n(), d, Xbuf.data(), XSqNorm.data());
  }

  void updateNormsForAppends(int_t oldN, int_t m) {
    if (!useNorms()) return;
    XSqNorm.resize(oldN + m);
    // compute norms only for appended portion
    array::sqNorm(m, d, Xbuf.data() + oldN*d, XSqNorm.data() + oldN);
  }

  double h;
  Kernel K;
  int_t d = 0;

  std::vector<T> Xbuf;
  std::vector<T> XSqNorm;

  int_t nextId = 0;
  std::unordered_map<int_t,int_t> idToSlot;
  std::vector<int_t> slotToId;
};

}

#endif // DEANN_DYNAMIC_NAIVE_KDE