#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <vector>
#include <queue>
#include <unordered_map>

// Edge data structure
struct Edge {
    int id;
    int v1, v2;
    double cost;
    Eigen::Vector3d optimal_pos;
    
    bool operator>(const Edge& other) const {
        return cost > other.cost;
    }
};

class QEMSimplifier {
private:
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<Eigen::Matrix4d> Q; // Quadric error matrix for each vertex
    std::vector<std::set<int>> vertex_faces; // Adjacent faces for each vertex
    std::vector<std::set<int>> vertex_vertices; // Adjacent vertices for each vertex
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> heap;
    std::unordered_map<long long, int> edge_id_map; // Map from edge to its id in heap
    std::vector<bool> vertex_deleted;
    std::vector<bool> face_deleted;
    int current_face_num;
    
    // Convert two vertex IDs to a unique edge ID
    long long edge_key(int v1, int v2) {
        if (v1 > v2) std::swap(v1, v2);
        return ((long long)v1 << 32) | v2;
    }
    
    // Compute quadric error matrix for a vertex
    void compute_vertex_quadric(int v) {
        Q[v] = Eigen::Matrix4d::Zero();
        
        // Iterate through all adjacent faces of the vertex
        for (int f : vertex_faces[v]) {
            if (face_deleted[f]) continue;
            
            // Get three vertices of the face
            Eigen::Vector3d p0 = V.row(F(f, 0));
            Eigen::Vector3d p1 = V.row(F(f, 1));
            Eigen::Vector3d p2 = V.row(F(f, 2));
            
            // Compute face normal
            Eigen::Vector3d n = ((p1 - p0).cross(p2 - p0)).normalized();
            
            // Plane equation: ax + by + cz + d = 0
            double a = n(0);
            double b = n(1);
            double c = n(2);
            double d = -n.dot(p0);
            
            // Build quadric matrix for this plane
            Eigen::Vector4d p(a, b, c, d);
            Q[v] += p * p.transpose();
        }
    }
    
    // Compute edge collapse cost and optimal position
    void compute_edge_cost(int v1, int v2, double& cost, Eigen::Vector3d& optimal_pos) {
        // Merge quadric error matrices of two vertices
        Eigen::Matrix4d Q_bar = Q[v1] + Q[v2];
        
        // Build matrix for solving optimal position
        Eigen::Matrix3d A = Q_bar.block<3,3>(0,0);
        Eigen::Vector3d b = -Q_bar.block<3,1>(0,3);
        
        // Add constraint row to make matrix invertible
        Eigen::Matrix4d Q_hat = Q_bar;
        Q_hat.row(3) << 0, 0, 0, 1;
        
        // Try to solve for optimal position
        bool invertible = false;
        if (std::abs(Q_hat.determinant()) > 1e-10) {
            Eigen::Vector4d x = Q_hat.inverse() * Eigen::Vector4d(0, 0, 0, 1);
            if (x(3) != 0) {
                optimal_pos = x.head<3>() / x(3);
                invertible = true;
            }
        }
        
        // If not invertible, try edge endpoints and midpoint
        if (!invertible) {
            Eigen::Vector3d v1_pos = V.row(v1);
            Eigen::Vector3d v2_pos = V.row(v2);
            Eigen::Vector3d mid_pos = 0.5 * (v1_pos + v2_pos);
            
            // Calculate error for three candidate positions
            double cost1 = compute_error(v1_pos, Q_bar);
            double cost2 = compute_error(v2_pos, Q_bar);
            double cost_mid = compute_error(mid_pos, Q_bar);
            
            // Choose position with minimum error
            if (cost1 <= cost2 && cost1 <= cost_mid) {
                optimal_pos = v1_pos;
                cost = cost1;
            } else if (cost2 <= cost1 && cost2 <= cost_mid) {
                optimal_pos = v2_pos;
                cost = cost2;
            } else {
                optimal_pos = mid_pos;
                cost = cost_mid;
            }
        } else {
            // Use optimal position to calculate error
            cost = compute_error(optimal_pos, Q_bar);
        }
    }
    
    // Calculate error of a point under the quadric form
    double compute_error(const Eigen::Vector3d& v, const Eigen::Matrix4d& Q_matrix) {
        Eigen::Vector4d v_bar(v(0), v(1), v(2), 1);
        return v_bar.transpose() * Q_matrix * v_bar;
    }
    
    // Build initial data structures
    void build_adjacency() {
        vertex_faces.resize(V.rows());
        vertex_vertices.resize(V.rows());
        
        // Build vertex-face adjacency
        for (int f = 0; f < F.rows(); f++) {
            for (int i = 0; i < 3; i++) {
                vertex_faces[F(f, i)].insert(f);
            }
        }
        
        // Build vertex-vertex adjacency
        for (int f = 0; f < F.rows(); f++) {
            for (int i = 0; i < 3; i++) {
                int v1 = F(f, i);
                int v2 = F(f, (i + 1) % 3);
                vertex_vertices[v1].insert(v2);
                vertex_vertices[v2].insert(v1);
            }
        }
    }
    
public:
    QEMSimplifier(const Eigen::MatrixXd& V_in, const Eigen::MatrixXi& F_in) 
        : V(V_in), F(F_in), current_face_num(F_in.rows()) {
        
        vertex_deleted.resize(V.rows(), false);
        face_deleted.resize(F.rows(), false);
        Q.resize(V.rows());
        
        // Build adjacency relationships
        build_adjacency();
        
        // Compute initial quadric error matrix for each vertex
        for (int v = 0; v < V.rows(); v++) {
            compute_vertex_quadric(v);
        }
        
        // Compute initial cost for all edges
        int edge_id = 0;
        for (int v1 = 0; v1 < V.rows(); v1++) {
            for (int v2 : vertex_vertices[v1]) {
                if (v1 < v2) { // Avoid duplicates
                    Edge e;
                    e.id = edge_id++;
                    e.v1 = v1;
                    e.v2 = v2;
                    compute_edge_cost(v1, v2, e.cost, e.optimal_pos);
                    heap.push(e);
                    edge_id_map[edge_key(v1, v2)] = e.id;
                }
            }
        }
    }
    
    // Perform one edge collapse
    bool collapse_edge() {
        while (!heap.empty()) {
            Edge e = heap.top();
            heap.pop();
            
            // Check if edge is still valid
            if (vertex_deleted[e.v1] || vertex_deleted[e.v2]) continue;
            
            // Check if this is the current edge (might have been updated)
            auto it = edge_id_map.find(edge_key(e.v1, e.v2));
            if (it == edge_id_map.end() || it->second != e.id) continue;
            
            // Perform edge collapse: keep v1, delete v2
            int v1 = e.v1;
            int v2 = e.v2;
            
            // Update position of v1
            V.row(v1) = e.optimal_pos;
            
            // Update quadric error matrix of v1
            Q[v1] = Q[v1] + Q[v2];
            
            // Transfer all adjacency information from v2 to v1
            for (int f : vertex_faces[v2]) {
                if (face_deleted[f]) continue;
                
                // Check if face is degenerate (contains both v1 and v2)
                bool degenerate = false;
                for (int i = 0; i < 3; i++) {
                    if (F(f, i) == v1) {
                        degenerate = true;
                        break;
                    }
                }
                
                if (degenerate) {
                    // Delete degenerate face
                    face_deleted[f] = true;
                    current_face_num--;
                } else {
                    // Update v2 to v1 in the face
                    for (int i = 0; i < 3; i++) {
                        if (F(f, i) == v2) {
                            F(f, i) = v1;
                            vertex_faces[v1].insert(f);
                        }
                    }
                }
            }
            
            // Update vertex adjacency
            for (int v : vertex_vertices[v2]) {
                if (v != v1 && !vertex_deleted[v]) {
                    vertex_vertices[v].erase(v2);
                    vertex_vertices[v].insert(v1);
                    vertex_vertices[v1].insert(v);
                    
                    // Recalculate edge cost
                    Edge new_edge;
                    new_edge.id = heap.size() + v; // Simple new id
                    new_edge.v1 = std::min(v1, v);
                    new_edge.v2 = std::max(v1, v);
                    compute_edge_cost(new_edge.v1, new_edge.v2, new_edge.cost, new_edge.optimal_pos);
                    heap.push(new_edge);
                    edge_id_map[edge_key(new_edge.v1, new_edge.v2)] = new_edge.id;
                }
            }
            
            // Delete v2
            vertex_deleted[v2] = true;
            vertex_faces[v2].clear();
            vertex_vertices[v2].clear();
            vertex_vertices[v1].erase(v2);
            
            // Recompute quadric error matrix for v1 (adjacent faces may have changed)
            compute_vertex_quadric(v1);
            
            return true;
        }
        
        return false;
    }
    
    // Get current mesh
    void get_mesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out) {
        // Build vertex mapping
        std::vector<int> vertex_map(V.rows(), -1);
        int new_vertex_count = 0;
        for (int v = 0; v < V.rows(); v++) {
            if (!vertex_deleted[v]) {
                vertex_map[v] = new_vertex_count++;
            }
        }
        
        // Copy vertices
        V_out.resize(new_vertex_count, 3);
        for (int v = 0; v < V.rows(); v++) {
            if (!vertex_deleted[v]) {
                V_out.row(vertex_map[v]) = V.row(v);
            }
        }
        
        // Copy faces
        F_out.resize(current_face_num, 3);
        int face_idx = 0;
        for (int f = 0; f < F.rows(); f++) {
            if (!face_deleted[f]) {
                for (int i = 0; i < 3; i++) {
                    F_out(face_idx, i) = vertex_map[F(f, i)];
                }
                face_idx++;
            }
        }
    }
    
    int get_face_count() { return current_face_num; }
};

int main(int argc, char * argv[])
{
    using namespace std;
    using namespace Eigen;
    
    cout << "QEM (Quadric Error Metrics) Mesh Simplification" << endl;
    cout << "  [space]  Perform one simplification step (remove 1% of faces)" << endl;
    cout << "  'r'      Reset to original mesh" << endl;
    cout << "  's'      Save current mesh" << endl;
    
    // Load mesh
    string filename("../3dmodels/frog.obj");
    if(argc >= 2) {
        filename = argv[1];
    }
    
    MatrixXd OV;
    MatrixXi OF;
    igl::read_triangle_mesh(filename, OV, OF);
    
    cout << "Original mesh: " << OV.rows() << " vertices, " << OF.rows() << " faces" << endl;
    
    igl::opengl::glfw::Viewer viewer;
    
    // QEM simplifier
    std::unique_ptr<QEMSimplifier> simplifier;
    MatrixXd V;
    MatrixXi F;
    
    // Reset function
    const auto reset = [&]() {
        simplifier = std::make_unique<QEMSimplifier>(OV, OF);
        simplifier->get_mesh(V, F);
        
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        viewer.data().set_face_based(true);
        
        cout << "Mesh reset" << endl;
    };
    
    // Perform one simplification step
    const auto decimate_step = [&]() {
        if (!simplifier) return;
        
        int target_collapses = std::max(1, (int)(simplifier->get_face_count() * 0.01));
        int collapsed = 0;
        
        for (int i = 0; i < target_collapses; i++) {
            if (simplifier->collapse_edge()) {
                collapsed++;
            } else {
                break;
            }
        }
        
        if (collapsed > 0) {
            simplifier->get_mesh(V, F);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.data().set_face_based(true);
            
            cout << "Collapsed " << collapsed << " edges, remaining faces: " << F.rows() << endl;
        } else {
            cout << "No more edges to collapse" << endl;
        }
    };
    
    // Keyboard callback
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod) -> bool {
        switch(key) {
            case ' ':
                decimate_step();
                return true;
            case 'r':
            case 'R':
                reset();
                return true;
            default:
                return false;
        }
    };
    
    // Initialize
    reset();
    viewer.core().is_animating = false;
    viewer.launch();
    
    return 0;
}