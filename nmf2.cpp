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
#include "nmf.hpp"

DEFINE_string(graph, "", "path to user item rating file, in snap format.");
DEFINE_int32(max_iter, 10000, "max iteration.");
DEFINE_int32(strip_width, 1024, "strip_width");

double NMF::MAXVAL = 1e+100;
double NMF::MINVAL = -1e+100;
NMF::Ftype NMF::px;
NMF::Ftype NMF::px2;


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

    /* update user side and item side */
    // 1. acc f
    NMF::reset_f(NMF::px);
    mapreduce_vec<NMF::Ftype, NMF::Ftype>(*f_item, NMF::px, NMF::acc_f);
    NMF::reset_f(NMF::px2);
    mapreduce_vec<NMF::Ftype, NMF::Ftype>(*f_user, NMF::px2, NMF::acc_f);

    //LOG(INFO) << " px : " << NMF::px.to_string();
    // 2. edge apply, gen update 
    graph->edge_apply<NMF::Ftype, NMF::Ftype, NMF::Ftype, NMF::Ftype>
      (*f_user,
       *f_item,
       *r_user,
       *r_item,
       NMF::acc_delta2);
    // 3. apply update

    binary_app<NMF::Ftype, NMF::Ftype, NMF::Ftype>(*f_user, *r_user, NMF::px, NMF::apply_delta2);
    binary_app<NMF::Ftype, NMF::Ftype, NMF::Ftype>(*f_item, *r_item, NMF::px2, NMF::apply_delta2);

    /* accumulate rmse */
    double rmse = 0.0;
    graph->edge_apply<NMF::Ftype, NMF::Ftype, double>
        (*f_user, *f_item, rmse, NMF::acc_error);
    rmse = std::sqrt(rmse / double(num_edges) );

    LOG(INFO) << " NMF::iterator " << iter << " rmse : " << rmse << " end .";

  }

  return 0;
}








