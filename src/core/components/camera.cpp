#include "camera.hpp"

void Camera::setProjectionMatrix(float fovInDegrees, float aspectRatio, float nearPlane, float farPlane) {
	this->aspectRatio = aspectRatio;
	this->fov = fovInDegrees;
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;
	
	projectionMatrix = matrix_perspective_right_hand(
		fov * (M_PI / 180.0f),
		aspectRatio,
		nearPlane,
		farPlane
	);
}

void Camera::updateCameraVectors() {
	simd::float3 newFront;
	newFront.x = cos(yaw * M_PI / 180.0f) * cos(pitch * M_PI / 180.0f);
	newFront.y = sin(pitch * M_PI / 180.0f);
	newFront.z = sin(yaw * M_PI / 180.0f) * cos(pitch * M_PI / 180.0f);
	
	front = simd::normalize(newFront);
	right = simd::normalize(simd::cross(front, worldUp));
	up = simd::normalize(simd::cross(right, front));
	
	// Update view matrix after camera vectors change
	setViewMatrix();
}

void Camera::processKeyboardInput(GLFWwindow* window, float deltaTime) {
	float velocity = movementSpeed * deltaTime;
	bool moved = false;
	
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        velocity = movementSpeed * deltaTime * 3;
    }
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position += front * velocity;
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position -= front * velocity;
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position -= right * velocity;
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position += right * velocity;
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		position += up * velocity;
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		position -= up * velocity;
		moved = true;
	}
	
	// Update view matrix only if the camera moved
	if (moved) {
		setViewMatrix();
	}
	
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}


void Camera::processMouseButton(GLFWwindow* window, int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;

            // Store the current cursor position as the last position
            glfwGetCursorPos(window, &lastX, &lastY);
        } else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

void Camera::processMouseMovement(float xpos, float ypos) {
    if (!isDragging) return;
    
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;
    
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;
        
    updateCameraVectors();
}

void Camera::setFrustumCornersWorldSpace(simd::float3* frustumCorners, float nearZ, float farZ) {
	const auto inv = matrix_invert(matrix_multiply(projectionMatrix, viewMatrix));
	int cornerIndex = 0;
	
	for (unsigned int x = 0; x < 2; x++) {
		for (unsigned int y = 0; y < 2; y++) {
			for (unsigned int z = 0; z < 2; z++) {
				// Create homogeneous coordinates
				simd::float4 pt = matrix_multiply(inv, simd::float4{
					2.0f * x - 1.0f,
					2.0f * y - 1.0f,
					z == 0 ? nearZ : farZ,
					1.0f
				});
				
				// Perform perspective division
				float invW = 1.0f / pt.w;
				frustumCorners[cornerIndex++] = simd::float3{
					pt.x * invW,
					pt.y * invW,
					pt.z * invW
				};
			}
		}
	}
}
