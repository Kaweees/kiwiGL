#pragma once

#include <vector>

#include "../include/vector3d.hpp"

namespace graphics {

class Mesh {
  public:
  // Constructor
  Mesh();

  // Destructor
  ~Mesh();

  // Method to load mesh data from a file
  bool loadFromFile(const std::string& filename);

  // Method to get vertices
  const std::vector<Vector3D>& getVertices() const;

  // Method to add a vertex
  void addVertex(const Vector3D& vertex);

  private:
  std::vector<Vector3D> vertices;  // Store vertices of the mesh
};

}  // namespace graphics
