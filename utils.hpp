#pragma once

#include <random>
#include <algorithm>


class RandGen {
    public :
    std::mt19937 gen;
    RandGen() {
      gen.seed(time(0));
    }

    double get_rand() {
      return std::uniform_real_distribution<> (-10.0, 10.0)(gen);
    }
};
