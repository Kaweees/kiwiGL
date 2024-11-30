#pragma once

#include <fstream>
#include <iostream>
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
        while (std::getline(file, line) && !file.eof()) {
          if (line.find("v ") != std::string::npos) {
            // Parse the vertex
            Vector3D vertex;
            if (sscanf(line.c_str(), "v %lf %lf %lf", &vertex.x, &vertex.y, &vertex.z) == 3) { addVertex(vertex); }
          } else if (line.find("vt ") != std::string::npos) {
            // Parse the texture coordinate
            Texture2D texture;
            if (sscanf(line.c_str(), "vt %lf %lf", &texture.u, &texture.v) == 2) { addTexture(texture); }
          } else if (line.find("f ") != std::string::npos) {
            // Parse the face
            int v1, v2, v3, t1, t2, t3, n1, n2, n3;
            if (sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3) ==
                9) {
              addFace(v1, v2, v3, textures[t1 - 1], textures[t2 - 1], textures[t3 - 1], WHITE);
            }
          }
        }
        file.close();
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
