#include <igl/circulation.h>
#include "../include/collapse_least_cost_edge.h"
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/decimate_trivial_callbacks.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>


void shortest_edge_midpoint(
  const int e,
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & /*F*/,
  const Eigen::MatrixXi & E,
  const Eigen::VectorXi & /*EMAP*/,
  const Eigen::MatrixXi & /*EF*/,
  const Eigen::MatrixXi & /*EI*/,
  double & cost,
  Eigen::RowVectorXd & p)
{
  cost = (V.row(E(e,0))-V.row(E(e,1))).norm();
  p = 0.5*(V.row(E(e,0))+V.row(E(e,1)));
}


int main(int argc, char * argv[])
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;

  cout<<"  [space]  perform one decimation step."<<endl;
  cout<<"  'r'  reset."<<endl;
  // Load a closed manifold mesh
  string filename("../3dmodels/frog.obj");
  if(argc>=2)
  {
    filename = argv[1];
  }
  MatrixXd V,OV;
  MatrixXi F,OF;
  read_triangle_mesh(filename,OV,OF);

  igl::opengl::glfw::Viewer viewer;

  // Prepare array-based edge data structures and priority queue
  VectorXi EMAP;
  MatrixXi E,EF,EI;
  igl::min_heap< std::tuple<double,int,int> > Q;
  Eigen::VectorXi EQ;
  // If an edge were collapsed, we'd collapse it to these points:
  MatrixXd C;
  int num_collapsed;

  // Function to reset original mesh and data structures
  const auto & reset = [&]()
  {
    F = OF;
    V = OV;
    edge_flaps(F,E,EMAP,EF,EI);
    C.resize(E.rows(),V.cols());
    VectorXd costs(E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    Q = {};
    EQ = Eigen::VectorXi::Zero(E.rows());
    {
      Eigen::VectorXd costs(E.rows());
      igl::parallel_for(E.rows(),[&](const int e)
      {
        double cost = e;
        RowVectorXd p(1,3);
        shortest_edge_midpoint(e,V,F,E,EMAP,EF,EI,cost,p);
        C.row(e) = p;
        costs(e) = cost;
      },10000);
      for(int e = 0;e<E.rows();e++)
      {
        Q.emplace(costs(e),e,0);
      }
    }

    num_collapsed = 0;
    viewer.data().clear();
    viewer.data().set_mesh(V,F);
    viewer.data().set_face_based(true);
  };

  // Function to perform one decimation step
  const auto & decimate_once = [&]()
  {
    if(!Q.empty())
    {
      bool something_collapsed = false;
      // collapse edge - only collapse 1% of edges per step
      const int max_iter = std::ceil(0.01*Q.size());
      for(int j = 0;j<max_iter;j++)
      {
        igl::decimate_pre_collapse_callback always_try;
        igl::decimate_post_collapse_callback never_care;
        igl::decimate_trivial_callbacks(always_try,never_care);
        // Explicit template instanciations expect std::function not raw pointer
        // Only relevant if IGL_STATIC_LIBRARY is defined
        const std::function<void (int, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, double&, Eigen::Matrix<double, 1, -1, 1, 1, -1>&)> cp = shortest_edge_and_midpoint;
        int e,e1,e2,f1,f2;
        if(!collapse_least_cost_edge(
              cp,always_try,never_care,
              V,F,E,
              EMAP,EF,EI,
              Q,EQ,C,
              e,e1,e2,f1,f2))
        {
          break;
        }
        something_collapsed = true;
        num_collapsed++;
      }

      if(something_collapsed)
      {
        viewer.data().clear();
        viewer.data().set_mesh(V,F);
        viewer.data().set_face_based(true);
        cout << "Collapsed edges: " << num_collapsed << ", Remaining faces: " << F.rows() << endl;
      }
    }
    else
    {
      cout << "No more edges to collapse!" << endl;
    }
  };

  const auto &pre_draw = [&](igl::opengl::glfw::Viewer & viewer)->bool
  {
    // Remove automatic animation - do nothing in pre_draw
    return false;
  };

  const auto &key_down =
    [&](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
  {
    switch(key)
    {
      case ' ':
        // Perform one decimation step when space is pressed
        decimate_once();
        break;
      case 'R':
      case 'r':
        reset();
        cout << "Mesh reset to original state" << endl;
        break;
      default:
        return false;
    }
    return true;
  };

  reset();
  viewer.core().background_color.setConstant(1);
  viewer.core().is_animating = false; // Disable automatic animation
  viewer.callback_key_down = key_down;
  viewer.callback_pre_draw = pre_draw;
  return viewer.launch();
}