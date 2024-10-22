#include "../include/mesh.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace graphics {

Mesh::Mesh() {}

Mesh::~Mesh() {}

bool Mesh::loadFromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open mesh file: " << filename << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double x, y, z;
    if (iss >> x >> y >> z) {
      addVertex(Vector3D(x, y, z));
    }
  }

  file.close();
  return true;
}

const std::vector<Vector3D>& Mesh::getVertices() const { return vertices; }

void Mesh::addVertex(const Vector3D& vertex) { vertices.push_back(vertex); }

}  // namespace graphics
