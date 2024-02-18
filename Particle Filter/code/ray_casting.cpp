//  16-833 SLAM_HW_Particle_Filter
//  Ray Casting c++ implementation
//  By Jeremy Kilbride and Ryan Wu

#include <iostream>
#include <vector>
#include <cmath>
#include "ray_casting.hpp"


std::vector<std::vector<double> > SensorModel::ray_casting(const std::vector<std::vector<double> >& x_t1, const std::vector<std::vector<double> >& map) {
    double grid_size = 10;
    size_t num_particles = x_t1.size();
    int subsample = 2;
    std::vector<std::vector<double> > z_t1_arr_star(num_particles, std::vector<double>(static_cast<int>(180 / subsample), 0.0));

    for (std::size_t particle = 0; particle < num_particles; ++particle) {
        double dir = x_t1[particle][2];
        double current_dir = dir - M_PI / 2;
        std::vector<double> start_point(2);
        start_point[0] = x_t1[particle][0];
        start_point[1] = x_t1[particle][1];

        for (int beam = 0; beam < static_cast<int>(180 / subsample); ++beam) {
            int signY_dir = 1;
            int signX_dir = 1;
            if (cos(current_dir) < 0) {
                signX_dir = -1;
            }
            if (sin(current_dir) < 0) {
                signY_dir = -1;
            }

            int grid_x = static_cast<int>(start_point[0] / grid_size);
            int grid_y = static_cast<int>(start_point[1] / grid_size);
            std::vector<double> current_point = start_point;
            double current_dist = 0;

            while (map[grid_y][grid_x] <= 0.01) {
                double dtX = ((grid_x)*grid_size - current_point[0]) / cos(current_dir);
                double dtY = ((grid_y)*grid_size - current_point[1]) / sin(current_dir);
                double dt = std::min(dtX, dtY);
                std::vector<double> next_point(2);
                next_point[0] = current_point[0] + dt * cos(current_dir);
                next_point[1] = current_point[1] + dt * sin(current_dir);
                current_dist += dt;
                current_point = next_point;
                if (dtX < dtY) {
                    grid_x += signX_dir;
                } else {
                    grid_y += signY_dir;
                }
                if (grid_x >= 800 || grid_y >= 800 || grid_x < 0 || grid_y < 0 || map[grid_y][grid_x] == -1) {
                    break;
                }
            }
            if (current_dist == 0) {
                current_dist = 5;
            }
            z_t1_arr_star[particle][beam] = current_dist;
            current_dir += M_PI / 180 * subsample;
        }
    }
    return z_t1_arr_star;
}