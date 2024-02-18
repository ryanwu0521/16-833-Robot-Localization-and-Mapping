#ifndef RAY_CASTING_HPP
#define RAY_CASTING_HPP

#include <vector>

class SensorModel {
public:
    SensorModel() {}

    SensorModel(const std::vector<std::vector<double> >& map) {
        // Initialize SensorModel with the provided map
    }

    std::vector<std::vector<double> > ray_casting(const std::vector<std::vector<double> >& x_t1, const std::vector<std::vector<double> >& map);
};

#endif