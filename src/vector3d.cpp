#include "../include/vector3d.hpp"

#include <cmath>

#include "../include/constants.hpp"
namespace graphics {
Vector2D Vector3D::project() const { return Vector2D(x * FOV, y * FOV); }

void Vector3D::rotateX(double theta) {
  double y_new = y * cos(theta) - z * sin(theta);
  double z_new = y * sin(theta) + z * cos(theta);
  y = y_new;
  z = z_new;
}

void Vector3D::rotateY(double theta) {
  double x_new = x * cos(theta) - z * sin(theta);
  double z_new = x * sin(theta) + z * cos(theta);
  x = x_new;
  z = z_new;
}

void Vector3D::rotateZ(double theta) {
  double x_new = x * cos(theta) - y * sin(theta);
  double y_new = x * sin(theta) + y * cos(theta);
  x = x_new;
  y = y_new;
}
}  // namespace graphics
