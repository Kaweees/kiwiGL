#pragma once

#include "../graphics/color.hpp"

namespace kiwigl {
  // constexpr int WIDTH = 640;
  constexpr int HEIGHT = 480;
  constexpr int FRAME_RATE = 60;
  constexpr int FRAME_TIME = 1000 / FRAME_RATE;
  constexpr int FOV = 640;
  const Color WHITE = Color(255, 255, 255, 255);
}  // namespace kiwigl
