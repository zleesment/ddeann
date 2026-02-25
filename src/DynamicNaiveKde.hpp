#ifndef DEANN_DYNAMIC_NAIVE_KDE
#define DEANN_DYNAMIC_NAIVE_KDE

#include "Kernel.hpp"
#include <vector>

namespace deann {

template<typename T>
class DynamicNaiveKde {
public:
    DynamicNaiveKde(double bandwidth, Kernel kernel);

private:
    double h;
    Kernel K;
    std::vector<T> data;
};

} // namespace deann

#endif // DEANN_DYNAMIC_NAIVE_KDE