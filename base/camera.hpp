/*
 * Basic camera class providing a look-at and first-person camera
 *
 * Copyright (C) 2016-2024 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera {
 private:
  float fov_;
  float znear_, zfar_;

  void updateViewMatrix() {
    glm::mat4 currentMatrix = matrices_.view;

    glm::mat4 rotM = glm::mat4(1.0f);
    glm::mat4 transM;

    rotM = glm::rotate(rotM, glm::radians(rotation_.x * (flipY ? -1.0f : 1.0f)),
                       glm::vec3(1.0f, 0.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(rotation_.y),
                       glm::vec3(0.0f, 1.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(rotation_.z),
                       glm::vec3(0.0f, 0.0f, 1.0f));

    glm::vec3 translation = position_;
    if (flipY) {
      translation.y *= -1.0f;
    }
    transM = glm::translate(glm::mat4(1.0f), translation);

    if (type_ == CameraType::firstperson) {
      matrices_.view = rotM * transM;
    } else {
      matrices_.view = transM * rotM;
    }

    viewPos_ = glm::vec4(position_, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

    if (matrices_.view != currentMatrix) {
      updated = true;
    }
  };

 public:
  enum CameraType { lookat, firstperson };
  CameraType type_ = CameraType::lookat;

  glm::vec3 rotation_ = glm::vec3();
  glm::vec3 position_ = glm::vec3();
  glm::vec4 viewPos_ = glm::vec4();

  float rotationSpeed = 1.0f;
  float movementSpeed = 1.0f;

  bool updated = true;
  bool flipY = false;

  struct {
    glm::mat4 perspective;
    glm::mat4 view;
  } matrices_;

  struct {
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
  } keys_;

  bool moving() const {
    return keys_.left || keys_.right || keys_.up || keys_.down;
  }

  float getNearClip() const { return znear_; }

  float getFarClip() const { return zfar_; }

  void setPerspective(float fov, float aspect, float znear, float zfar) {
    glm::mat4 currentMatrix = matrices_.perspective;
    this->fov_ = fov;
    this->znear_ = znear;
    this->zfar_ = zfar;
    matrices_.perspective =
        glm::perspective(glm::radians(fov), aspect, znear, zfar);
    if (flipY) {
      matrices_.perspective[1][1] *= -1.0f;
    }
    if (matrices_.view != currentMatrix) {
      updated = true;
    }
  };

  void updateAspectRatio(float aspect) {
    glm::mat4 currentMatrix = matrices_.perspective;
    matrices_.perspective =
        glm::perspective(glm::radians(fov_), aspect, znear_, zfar_);
    if (flipY) {
      matrices_.perspective[1][1] *= -1.0f;
    }
    if (matrices_.view != currentMatrix) {
      updated = true;
    }
  }

  void setPosition(glm::vec3 position) {
    this->position_ = position;
    updateViewMatrix();
  }

  void setRotation(glm::vec3 rotation) {
    this->rotation_ = rotation;
    updateViewMatrix();
  }

  void rotate(glm::vec3 delta) {
    this->rotation_ += delta;
    updateViewMatrix();
  }

  void setTranslation(glm::vec3 translation) {
    this->position_ = translation;
    updateViewMatrix();
  };

  void translate(glm::vec3 delta) {
    this->position_ += delta;
    updateViewMatrix();
  }

  void setRotationSpeed(float rotationSpeed) {
    this->rotationSpeed = rotationSpeed;
  }

  void setMovementSpeed(float movementSpeed) {
    this->movementSpeed = movementSpeed;
  }

  void update(float deltaTime) {
    updated = false;
    if (type_ == CameraType::firstperson) {
      if (moving()) {
        glm::vec3 camFront;
        camFront.x =
            -cos(glm::radians(rotation_.x)) * sin(glm::radians(rotation_.y));
        camFront.y = sin(glm::radians(rotation_.x));
        camFront.z =
            cos(glm::radians(rotation_.x)) * cos(glm::radians(rotation_.y));
        camFront = glm::normalize(camFront);

        float moveSpeed = deltaTime * movementSpeed;

        if (keys_.up)
          position_ += camFront * moveSpeed;
        if (keys_.down)
          position_ -= camFront * moveSpeed;
        if (keys_.left)
          position_ -= glm::normalize(
                           glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) *
                       moveSpeed;
        if (keys_.right)
          position_ += glm::normalize(
                           glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) *
                       moveSpeed;
      }
    }
    updateViewMatrix();
  };

  // Update camera passing separate axis data (gamepad)
  // Returns true if view or position has been changed
  bool updatePad(glm::vec2 axisLeft, glm::vec2 axisRight, float deltaTime) {
    bool retVal = false;

    if (type_ == CameraType::firstperson) {
      // Use the common console thumbstick layout
      // Left = view, right = move

      const float deadZone = 0.0015f;
      const float range = 1.0f - deadZone;

      glm::vec3 camFront;
      camFront.x =
          -cos(glm::radians(rotation_.x)) * sin(glm::radians(rotation_.y));
      camFront.y = sin(glm::radians(rotation_.x));
      camFront.z =
          cos(glm::radians(rotation_.x)) * cos(glm::radians(rotation_.y));
      camFront = glm::normalize(camFront);

      float moveSpeed = deltaTime * movementSpeed * 2.0f;
      float rotSpeed = deltaTime * rotationSpeed * 50.0f;

      // Move
      if (fabsf(axisLeft.y) > deadZone) {
        float pos = (fabsf(axisLeft.y) - deadZone) / range;
        position_ -=
            camFront * pos * ((axisLeft.y < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
        retVal = true;
      }
      if (fabsf(axisLeft.x) > deadZone) {
        float pos = (fabsf(axisLeft.x) - deadZone) / range;
        position_ +=
            glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) *
            pos * ((axisLeft.x < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
        retVal = true;
      }

      // Rotate
      if (fabsf(axisRight.x) > deadZone) {
        float pos = (fabsf(axisRight.x) - deadZone) / range;
        rotation_.y += pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
        retVal = true;
      }
      if (fabsf(axisRight.y) > deadZone) {
        float pos = (fabsf(axisRight.y) - deadZone) / range;
        rotation_.x -= pos * ((axisRight.y < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
        retVal = true;
      }
    } else {
      // todo: move code from example base class for look-at
    }

    if (retVal) {
      updateViewMatrix();
    }

    return retVal;
  }
};