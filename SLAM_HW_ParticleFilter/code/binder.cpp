#include <pybind11/pybind11.h>
#include "ray_casting.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

SensorModel create_sensor_model(const std::vector<std::vector<double>>& map) {
    return SensorModel(map);
}

PYBIND11_MODULE(ray_casting_lib, m) {
    py::class_<SensorModel>(m, "SensorModel")
        .def(py::init<>())
        .def("ray_casting", &SensorModel::ray_casting);

    m.def("create_sensor_model", &create_sensor_model);
    m.attr("__version__") = "0.0.1";
}