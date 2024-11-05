#pragma once

#include <vector>

#include "../include/face.hpp"
#include "../include/texture.hpp"
#include "../include/vector3d.hpp"
#include <string>

namespace graphics {

class Mesh {
  public:
  // Constructor to initialize memory
  Mesh();

  // Destructor to free the memory allocated
  ~Mesh() = default;

  // Method to load a mesh from a file
  void loadMesh(const std::string& filename);

  // Method to load textures from a Wavefront .obj file
  bool loadOBJ(const std::string& filename);

  // Method to load textures from a file
  bool loadTextures(const std::string& filename);

  // Method to get vertices
  const std::vector<Vector3D>& getVertices() const;

  // Method to add a vertex
  void addVertex(const Vector3D& vertex);

  // Method to get textures
  const std::vector<Texture2D>& getTextures() const;

  // Method to add a texture
  void addTexture(const Texture2D& texture);

  // Method to add a face
  void addFace(int v1, int v2, int v3, const Texture2D& t1, const Texture2D& t2,
      const Texture2D& t3, const Color& color);

  std::vector<Vector3D> vertices;
  std::vector<Face> faces;
  std::vector<Texture2D> textures;
  Vector3D scale;
  Vector3D rotation;
  Vector3D translation;
};

}  // namespace graphics
