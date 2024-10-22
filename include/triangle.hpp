#pragma once

class Triangle {
  public:
  Triangle(int v1, int v2, int v3) : vertexIndices{v1, v2, v3} {}

  // Method to get vertex indices
  const int* getVertexIndices() const { return vertexIndices; }

  private:
  int vertexIndices[3];  // Indices of the vertices that form the triangle
};
