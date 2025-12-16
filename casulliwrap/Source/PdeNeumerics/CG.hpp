#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace CFD {

template<typename T>
class Vector {
public:
    std::vector<T> data;

    Vector() = default;
    Vector(size_t n) : data(n, 0) {}
    Vector(const std::vector<T>& v) : data(v) {}
    size_t size() const { return data.size(); }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    void zero() { std::fill(data.begin(), data.end(), T(0)); }

    T dot(const Vector<T>& other) const {
        if(size() != other.size()) throw std::runtime_error("Vector size mismatch in dot");
        T sum = 0;
        for(size_t i=0; i<size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    Vector<T>& operator+=(const Vector<T>& other) {
        if(size() != other.size()) throw std::runtime_error("Vector size mismatch in +=");
        for(size_t i=0; i<size(); ++i) data[i] += other[i];
        return *this;
    }

    Vector<T>& operator-=(const Vector<T>& other) {
        if(size() != other.size()) throw std::runtime_error("Vector size mismatch in -=");
        for(size_t i=0; i<size(); ++i) data[i] -= other[i];
        return *this;
    }

    Vector<T> operator*(T scalar) const {
        Vector<T> result(*this);
        for(size_t i=0; i<size(); ++i) result[i] *= scalar;
        return result;
    }
};

template<typename T>
class Matrix {
public:
    std::vector<std::vector<T>> data;

    Matrix() = default;
    Matrix(size_t n, size_t m) : data(n, std::vector<T>(m, 0)) {}
    size_t rows() const { return data.size(); }
    size_t cols() const { return data.empty() ? 0 : data[0].size(); }

    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }

    Vector<T> multiply(const Vector<T>& x) const {
        if(cols() != x.size()) throw std::runtime_error("Matrix-vector size mismatch");
        Vector<T> result(rows());
        for(size_t i=0; i<rows(); ++i) {
            T sum = 0;
            for(size_t j=0; j<cols(); ++j) sum += data[i][j] * x[j];
            result[i] = sum;
        }
        return result;
    }
};

template<typename T>
Vector<T> conjugate_gradient(const Matrix<T>& A, const Vector<T>& b, T tol=1e-8, size_t max_iter=1000) {
    size_t n = b.size();
    Vector<T> x(n); // initial guess = 0
    Vector<T> r = b - A.multiply(x);
    Vector<T> p = r;
    T rsold = r.dot(r);

    if(rsold < tol*tol) return x;

    for(size_t i=0; i<max_iter; ++i) {
        Vector<T> Ap = A.multiply(p);
        T alpha = rsold / p.dot(Ap);
        x += p * alpha;
        r -= Ap * alpha;

        T rsnew = r.dot(r);
        if(rsnew < tol*tol) {
            std::cout << "CG converged in " << i+1 << " iterations.\n";
            break;
        }

        T beta = rsnew / rsold;
        p = r + p * beta;
        rsold = rsnew;
    }

    return x;
}

} // namespace CFD
