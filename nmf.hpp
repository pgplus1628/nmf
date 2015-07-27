#pragma once
#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <algorithm>
#include <random>
#include <cmath>

#include "graph.hpp"
#include "vec.hpp"
#include "utils.hpp"

static RandGen RG;

class NMF {
    public :

    static const size_t NLATENT = 128;
    static constexpr double epsilon = 1e-16;
    static double MAXVAL;
    static double MINVAL;


    // ----------------/
    // Types
    // ----------------/

    struct Etype {
      float obs;
      friend std::istream & operator >> (std::istream& is, struct Etype &eval){
        is >> eval.obs;
        return is;
      }
      friend std::ostream & operator << (std::ostream& os, struct Etype &eval) {
        os << eval.obs;
        return os;
      }
    };
    typedef struct Etype Etype;

    struct Ftype {
      double pvec[NLATENT];
      std::string to_string() { 
        std::string ret;
        std::for_each(pvec, pvec + NLATENT, [&](double &x) {ret += " " + std::to_string(x) ;});
        return ret;
      }
    };
    typedef struct Ftype Ftype;


    static Ftype px;
    static Ftype px2;


    // -----------------/
    // Init Functions
    // -----------------/
    static void reset_f(Ftype &f) {
      for(size_t i = 0;i < NLATENT;i ++) {
        f.pvec[i] = 0.0;
      }
    }

    static void rand_f(Ftype &f) {
      for(size_t i = 0;i < NLATENT;i ++) {
        f.pvec[i] = RG.get_rand();
      }
    }

    static void check_px(Ftype &p){
      for(size_t i = 0;i < NLATENT;i ++) {
        CHECK_NE(p.pvec[i], 0.0);
      }
    }

    // ------------------/
    // NMF Core Functions
    // ------------------/

    static void acc_f(Ftype &f, Ftype &facc) {
      for(size_t i = 0;i < NLATENT;i ++) {
        facc.pvec[i] += f.pvec[i];
      }
    }


    static void acc_delta(Ftype &f1, Ftype &f2, Etype &e,
                          Ftype &r1)
    {
      double pred = 0.0;
      for(size_t i = 0;i < NLATENT;i ++) {
        pred += f1.pvec[i] * f2.pvec[i];
      }
      pred = std::min(pred, MAXVAL);
      pred = std::max(pred, MINVAL);
      for(size_t i = 0;i < NLATENT;i ++) {
        r1.pvec[i] +=  f2.pvec[i] * e.obs / pred;
      }
    }

    static void acc_delta2(Ftype &f1, Ftype &f2, Etype &e, 
                           Ftype &r1, Ftype &r2)
    {
      double pred = 0.0;
      for(size_t i = 0;i < NLATENT;i ++) {
        pred += f1.pvec[i] * f2.pvec[i];
      }

      pred = std::min(pred, MAXVAL);
      pred = std::max(pred, MINVAL);
      for(size_t i = 0;i < NLATENT;i ++) {
        r1.pvec[i] += f2.pvec[i] * e.obs / pred;
        r2.pvec[i] += f1.pvec[i] * e.obs / pred;
      }
    }

    static void apply_delta(Ftype &f, Ftype &r)
    {
      for(size_t i = 0;i < NLATENT;i ++) {
        f.pvec[i] *= r.pvec[i] / px.pvec[i];
        if (f.pvec[i] < epsilon)
          f.pvec[i] = epsilon;
      }
    }

    
    static void apply_delta2(Ftype &f, Ftype &r, Ftype &p)
    {
      for(size_t i = 0;i < NLATENT;i ++) {
        f.pvec[i] *= r.pvec[i] / p.pvec[i];
        if (f.pvec[i] < epsilon)
          f.pvec[i] = epsilon;
      }
    }


    static void acc_error(Ftype &f_user, Ftype &f_item, Etype &e, double &rmse)
    {
      double pred = 0.0;
      for(size_t i = 0;i < NLATENT;i ++) {
        pred += f_user.pvec[i] * f_item.pvec[i];
      }
      pred = std::min(pred, MAXVAL);
      pred = std::max(pred, MINVAL);
      double err = e.obs - pred;
      rmse += err * err;
    }


};


