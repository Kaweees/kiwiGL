#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../core/constants.hpp"
#include "../geometry/face.hpp"
#include "../geometry/vector.hpp"
#include "../graphics/color.hpp"
#include "../texture/texture.hpp"

namespace kiwigl {

class Mesh {
  public:
    // Constructor to initialize memory
    Mesh() = default;

    // Destructor to free the memory allocated
    ~Mesh() = default;

    // Method to load a mesh from a file
    void loadMesh(const std::string& filename) {
      loadOBJ(filename);
      // loadTexture(filename);
    }

    // Method to load textures from a Wavefront .obj file
    bool loadOBJ(const std::string& filename) {
      std::ifstream file(filename, std::ios::binary);
      if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
          if (line.rfind("vt ", 0) == 0) {
            // Parse the texture coordinate
            Texture2D texture;
            if (sscanf(line.c_str(), "vt %lf %lf", &texture.u, &texture.v) == 2) { addTexture(texture); }
          } else if (line.rfind("v ", 0) == 0) {
            // Parse the vertex
            Vector3D vertex;
            if (sscanf(line.c_str(), "v %lf %lf %lf", &vertex.x, &vertex.y, &vertex.z) == 3) { addVertex(vertex); }
          } else if (line.rfind("f ", 0) == 0) {
            std::stringstream ss(line.substr(2));
            std::string token;
            std::vector<int> v_indices, t_indices, n_indices;

            while (ss >> token) {
              int v = 0, t = 0, n = 0;
              if (sscanf(token.c_str(), "%d/%d/%d", &v, &t, &n) == 3) {
                v_indices.push_back(v);
                t_indices.push_back(t);
                n_indices.push_back(n);
              } else if (sscanf(token.c_str(), "%d//%d", &v, &n) == 2) {
                v_indices.push_back(v);
                n_indices.push_back(n);
              } else if (sscanf(token.c_str(), "%d/%d", &v, &t) == 2) {
                v_indices.push_back(v);
                t_indices.push_back(t);
              } else if (sscanf(token.c_str(), "%d", &v) == 1) {
                v_indices.push_back(v);
              }
            }

            // Triangulate polygon faces
            if (v_indices.size() >= 3) {
              for (size_t i = 1; i < v_indices.size() - 1; ++i) {
                Texture2D tex[3];
                if (t_indices.size() == v_indices.size()) {
                  auto get_abs_idx = [](int idx, size_t s) -> size_t {
                    if (idx > 0) return idx - 1;
                    if (idx < 0) return s + idx;
                    return (size_t)-1; // Invalid index
                  };

                  size_t t0 = get_abs_idx(t_indices[0], textures.size());
                  size_t ti = get_abs_idx(t_indices[i], textures.size());
                  size_t ti1 = get_abs_idx(t_indices[i + 1], textures.size());

                  if (t0 < textures.size() && ti < textures.size() && ti1 < textures.size()) {
                    tex[0] = textures[t0];
                    tex[1] = textures[ti];
                    tex[2] = textures[ti1];
                  }
                }

                addFace(v_indices[0], v_indices[i], v_indices[i + 1], tex[0], tex[1], tex[2], WHITE);
              }
            }
          }
        }
        file.close();

        // Center the mesh to its origin
        if (!vertices.empty()) {
          Vector3D center(0, 0, 0);
          for (const auto& v : vertices) {
            center.x += v.x;
            center.y += v.y;
            center.z += v.z;
          }
          center.x /= vertices.size();
          center.y /= vertices.size();
          center.z /= vertices.size();

          for (auto& v : vertices) {
            v.x -= center.x;
            v.y -= center.y;
            v.z -= center.z;
          }
        }

        return true;
      } else {
        std::cerr << "Failed to open mesh file: " << filename << std::endl;
        return false;
      }
    }

    // Method to load textures from a file
    bool loadTextures(const std::string& filename);

    // Method to get vertices
    const std::vector<Vector3D>& getVertices() const { return vertices; }

    // Method to add a vertex
    void addVertex(const Vector3D& vertex) { vertices.push_back(vertex); }

    // Method to get textures
    const std::vector<Texture2D>& getTextures() const { return textures; }

    // Method to add a texture
    void addTexture(const Texture2D& texture) { textures.push_back(texture); }

    // Method to add a face
    void addFace(int v1, int v2, int v3, const Texture2D& t1, const Texture2D& t2, const Texture2D& t3,
                 const Color& color) {
      faces.push_back(Face(v1, v2, v3, t1, t2, t3, color));
    }

    std::vector<Vector3D> vertices;
    std::vector<Face> faces;
    std::vector<Texture2D> textures;
    Vector3D scale;
    Vector3D rotation;
    Vector3D translation;
};

} // namespace kiwigl
