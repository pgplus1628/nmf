#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <random>
#include <cmath>

#include "graph.hpp"
#include "vec.hpp"
#include "utils.hpp"

DEFINE_string(graph, "", "path to user item rating file, in snap format.");
DEFINE_int32(max_iter, 10000, "max iteration.");
DEFINE_int32(strip_width, 1024, "strip_width");

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

    static void check_px(){
      for(size_t i = 0;i < NLATENT;i ++) {
        CHECK_NE(px.pvec[i], 0.0);
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
        pred += 
        f1.pvec[i] * 
        f2.pvec[i];
      }
      pred = std::min(pred, MAXVAL);
      pred = std::max(pred, MINVAL);
      for(size_t i = 0;i < NLATENT;i ++) {
        r1.pvec[i] +=  
        f2.pvec[i] 
        * e.obs / pred;
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


    static void acc_error(Ftype &f_user, Ftype &f_item, Etype &e, double &rmse)
    {
      double pred = 0.0;
      for(size_t i = 0;i < NLATENT;i ++) {
        pred += f_user.pvec[i] * f_item.pvec[i];
      }
      pred = std::min(pred, MAXVAL);
      pred = std::max(pred, MINVAL);
      float err = e.obs - pred;
      rmse += err * err;
    }


};

double NMF::MAXVAL = 1e+100;
double NMF::MINVAL = -1e+100;
NMF::Ftype NMF::px;


int main(int argc, char ** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  Graph<NMF::Etype> * graph = new Graph<NMF::Etype>(FLAGS_strip_width);
  graph->load(FLAGS_graph);
  LOG(INFO) << " Graph Load Finished.";

  size_t u_len = graph->get_dim().first;
  size_t v_len = graph->get_dim().second;
  size_t num_edges = graph->get_num_edges();

  LOG(INFO) << " graph dim : " << u_len  << " , " << v_len;

  std::vector<NMF::Ftype> *f_user = new std::vector<NMF::Ftype>(u_len);
  std::vector<NMF::Ftype> *f_item = new std::vector<NMF::Ftype>(v_len);

  std::vector<NMF::Ftype> *r_user = new std::vector<NMF::Ftype>(u_len);
  std::vector<NMF::Ftype> *r_item = new std::vector<NMF::Ftype>(v_len);


  /* train */
  unary_app<NMF::Ftype>(*f_user, NMF::rand_f);
  unary_app<NMF::Ftype>(*f_item, NMF::rand_f);

  LOG(INFO) << " Begin Main Loop .";

  for(size_t iter = 0; iter < FLAGS_max_iter; iter ++) {
    unary_app<NMF::Ftype>(*r_user, NMF::reset_f);
    unary_app<NMF::Ftype>(*r_item, NMF::reset_f);

    /* update user side */
    // 1. acc item f
    NMF::reset_f(NMF::px);
    mapreduce_vec<NMF::Ftype, NMF::Ftype>(*f_item, NMF::px, NMF::acc_f);
    //LOG(INFO) << " px : " << NMF::px.to_string();
    // 2. edge apply, gen update 
    graph->edge_apply<NMF::Ftype, NMF::Ftype, NMF::Ftype>
        (*f_user,
         *f_item,
         *r_user,
         NMF::acc_delta,
         false);
    // 3. apply update
    binary_app<NMF::Ftype, NMF::Ftype>(*f_user, *r_user, NMF::apply_delta);


    //dump_vec<NMF::Ftype>(*r_user, "r_user_" + std::to_string(iter) + ".dat");
    //dump_vec<NMF::Ftype>(*f_user, "f_user_" + std::to_string(iter) + ".dat");


    /* update item side */
    // 1. acc user f
    NMF::reset_f(NMF::px);
    mapreduce_vec<NMF::Ftype, NMF::Ftype>(*f_user, NMF::px, NMF::acc_f);
    //LOG(INFO) << " px : " << NMF::px.to_string();
    // 2. edge apply, gen update
    graph->edge_apply<NMF::Ftype, NMF::Ftype, NMF::Ftype>
        (*f_item,
         *f_user,
         *r_item,
         NMF::acc_delta,
         true);
    // 3. apply update
    binary_app<NMF::Ftype, NMF::Ftype>(*f_item, *r_item, NMF::apply_delta);
    //dump_vec<NMF::Ftype>(*f_user, "f_item_" + std::to_string(iter) + ".dat");

    /* accumulate rmse */
    double rmse = 0.0;
    graph->edge_apply<NMF::Ftype, NMF::Ftype, double>
        (*f_user, *f_item, rmse, NMF::acc_error);
    rmse = std::sqrt(rmse / double(num_edges) );

    LOG(INFO) << " NMF::iterator " << iter << " rmse : " << rmse << " end .";

  }

  return 0;
}








