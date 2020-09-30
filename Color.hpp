#pragma once

struct Color {
    float x;
    float y;
    float z;

    Color& operator+=(const Color& other) {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    __device__ friend Color operator+(const Color& left, const Color& right) {
        return Color{left.x + right.x, left.y + right.y, left.z + right.z};
    }
};
