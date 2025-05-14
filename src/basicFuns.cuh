#pragma once
#include "ghostCellPresets.cuh"
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
using thrust::device_vector;
using thrust::host_vector;
using thrust::max;
using thrust::min;
using thrust::reduce;
using thrust::swap;

// flux function at a cell interface
__device__ vec5d flux(const vec5d &U, const vec5d &U_t, double k)
{
    double rho = U(0);
    double pi = U(1);
    double u = U(2);
    double v = U(3);
    double e = U(4);

    double rho_t = U_t(0);
    double pi_t = U_t(1);
    double u_t = U_t(2);
    double v_t = U_t(3);
    double e_t = U_t(4);

    return {
        rho * u + k / 2.0 * (rho * u_t + u * rho_t),
        rho * u * u + pi + k / 2.0 * (rho_t * u * u + 2.0 * rho * u * u_t + pi_t),
        rho * u * (e + u * u / 2.0 + v * v / 2.0) + pi * u + k / 2.0 * (rho_t * u * e + rho * u_t * e + rho * u * e_t + rho_t * u * u * u / 2.0 + 1.5 * rho * u * u * u_t + rho_t * u * v * v / 2.0 + rho * u * v * v_t + rho * u_t * v * v / 2.0 + pi_t * u + pi * u_t),
        rho * u * v + k / 2.0 * (rho * u_t * v + u * rho_t * v + u * rho * v_t),
        0.0};
}

// MUSCL reconstruction procedure
__global__ void MUSCL_rel_RP_GRP(vec5d *U, vec5d *Us, int Nx, int Ny, double alpha, double width, vec5d *U_lr_interface)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-index
    double h = width / double(Nx);

    if (i < Nx - 1 && j < Ny && i > 0)
    {
        int idxL = j * Nx + i - 1; // left cell
        int idx = j * Nx + i;      // cell i
        int idxR = j * Nx + i + 1; // Right cell

        vec5d Ul_s = (U[idxR] - U[idxL]) * 0.5 * (1.0 / h);
        vec5d U_s = (U[idx] - U[idxL]) * (1.0 / h) * alpha;
        vec5d Ur_s = (U[idxR] - U[idx]) * (1.0 / h) * alpha;
        for (auto i = 0; i <= 3; i++)
        {
            double sgn13 = Ul_s(i) * Ur_s(i), sgn23 = U_s(i) * Ur_s(i);
            if (sgn13 > 0.0 && sgn23 > 0.0)
            {
                double sgn = 1.0;
                if (Ul_s(i) <= 0.0)
                {
                    sgn = -1.0;
                }
                Us[idx][i] = sgn * min(abs(Ul_s(i)), min(abs(U_s(i)), abs(Ur_s(i))));
            }
            else
            {
                Us[idx][i] = 0.0;
            }
        }
        U_lr_interface[2 * idx + 1] = U[idx] - Us[idx] * h * 0.5;
        U_lr_interface[2 * idx + 2] = U[idx] + Us[idx] * h * 0.5;
    }
}

// compute the approximate Riemann solutions and correct the negativity
__global__ void rel_RP_posi_fix(vec5d *U, vec5d *Us, vec5d *U_lr_interface, int Nx, int Ny, vec5d *u_u_t_interface, int *c_lr_interface_edited, double gamma)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-index

    if (i < Nx - 1 && j < Ny && i > 1)
    {
        int idxL = j * Nx + i - 1; // left cell
        int idx = j * Nx + i;      // cell i(left interface)
        int idxR = j * Nx + i + 1; // Right cell

        double rho_l = U_lr_interface[2 * idx](0);
        double rho_r = U_lr_interface[2 * idx + 1](0);
        double pi_l = U_lr_interface[2 * idx](1);
        double pi_r = U_lr_interface[2 * idx + 1](1);
        double u_l = U_lr_interface[2 * idx](2);
        double u_r = U_lr_interface[2 * idx + 1](2);
        double v_l = U_lr_interface[2 * idx](3);
        double v_r = U_lr_interface[2 * idx + 1](3);
        double e_l = pi_l / rho_l / (gamma - 1.0);
        double e_r = pi_r / rho_r / (gamma - 1.0);
        double c_l = 0.0;
        double c_r = 0.0;
        double rho_lx = Us[idxL](0);
        double rho_rx = Us[idx](0);
        double pi_lx = Us[idxL](1);
        double pi_rx = Us[idx](1);
        double u_lx = Us[idxL](2);
        double u_rx = Us[idx](2);
        double v_lx = Us[idxL](3);
        double v_rx = Us[idx](3);
        double e_lx = pi_lx / rho_l / (gamma - 1.0) - pi_l * rho_lx / rho_l / rho_l / (gamma - 1.0);
        double e_rx = pi_rx / rho_r / (gamma - 1.0) - pi_r * rho_rx / rho_r / rho_r / (gamma - 1.0);
        double c_lx = 0.5 * sqrt(gamma / pi_l / rho_l) * (pi_l * rho_lx + rho_l * pi_lx);
        double c_rx = 0.5 * sqrt(gamma / pi_r / rho_r) * (pi_r * rho_rx + rho_r * pi_rx);
        double rho = 0.0, pi = 0.0, u = 0.0, v = 0.0, e = 0.0;
        double rho_t = 0.0, pi_t = 0.0, u_t = 0.0, v_t = 0.0, e_t = 0.0;

        double a_l = sqrt(gamma * pi_l / rho_l), a_r = sqrt(gamma * pi_r / rho_r);
        double kappa = (gamma + 1.0) / 2.0;
        if (pi_r >= pi_l)
        {
            double c_l_ = rho_l * (a_l + kappa * max(0.0, (pi_r - pi_l) / rho_r / a_r + u_l - u_r));
            c_l = c_l_;
            c_r = rho_r * (a_r + kappa * max(0.0, (pi_l - pi_r) / c_l_ + u_l - u_r));
        }
        else
        {
            double c_r_ = rho_r * (a_r + kappa * max(0.0, (pi_l - pi_r) / rho_l / a_l + u_l - u_r));
            double c_l_ = rho_l * (a_l + kappa * max(0.0, (pi_r - pi_l) / c_r_ + u_l - u_r));
            c_l = c_l_;
            c_r = c_r_;
        }

        if (u_l - c_l / rho_l >= 0.0)
        {
            rho = rho_l, pi = pi_l, u = u_l, v = v_l, e = e_l;
            rho_t = -rho * u_lx - u * rho_lx;
            u_t = -u * u_lx - pi_lx / rho;
            e_t = -u * e_lx - pi / rho * u_lx;
            v_t = -u * v_lx;
            pi_t = -u * pi_lx - c_l * c_l / rho_l * u_lx;
        }
        else if (u_r + c_r / rho_r <= 0.0)
        {
            rho = rho_r, pi = pi_r, u = u_r, v = v_r, e = e_r;
            rho_t = -rho * u_rx - u * rho_rx;
            u_t = -u * u_rx - pi_rx / rho;
            e_t = -u * e_rx - pi / rho * u_rx;
            v_t = -u * v_rx;
            pi_t = -u * pi_rx - c_r * c_r / rho * u_rx;
        }
        else
        {
            double u_m = (u_l * c_l + u_r * c_r + pi_l - pi_r) / (c_l + c_r);
            double pi_m = ((u_l - u_r) * c_l * c_r + c_l * pi_r + c_r * pi_l) / (c_l + c_r);
            double k_l = -pi_lx - c_l * u_lx + 0.5 * (u - u_l) * c_lx;
            double k_r = -pi_rx + c_r * u_rx - 0.5 * (u - u_r) * c_rx;
            double d_l = -2.0 * c_l * c_l / rho_l * u_lx - 2.0 * c_l / rho_l * pi_lx + c_l / rho_l * (u - u_l) * c_lx;
            double d_r = 2.0 * c_r * c_r / rho_r * u_rx - 2.0 * c_r / rho_r * pi_rx + c_r / rho_r * (u_r - u) * c_rx;
            double Dpi_Dt = (c_r * d_l - c_l * d_r) / 2.0 / (c_l + c_r);
            pi = pi_m;
            u = u_m;
            if (u <= 0.0)
            {
                double rho_mr = c_r / (u_r - u_m + c_r / rho_r);
                double e_mr = e_r + (pi_m - pi_r) * (pi_m + pi_r) / 2.0 / c_r / c_r;
                rho = rho_mr;
                e = e_mr;
                v = v_r;
                rho_t = -u * rho * rho * rho / (rho_r * c_r * c_r) * (c_r * c_r / rho_r / rho_r * rho_rx - c_r * u_rx + 1.5 * (u - u_r) * c_rx) + rho * rho * rho / (c_r * c_r * c_r) * (u + c_r / rho) * Dpi_Dt;
                pi_t = 1.0 / (c_l + c_r) * (c_l * c_r / rho_l * (1 + u * rho / c_r) * k_l - c_l * c_r / rho_r * (1 - u * rho / c_l) * k_r);
                u_t = 1.0 / (c_l + c_r) * (c_l / rho_l * (1 + u * rho / c_r) * k_l + c_r / rho_r * (1 - u * rho * c_l / c_r / c_r) * k_r);
                e_t = -u * rho / rho_r / c_r / c_r / c_r * (-pi_r * c_r * pi_rx + c_r * c_r * c_r * e_rx + 2.0 * (e_r - e) * c_r * c_r * c_rx) + pi / c_r / c_r * pi_t;
                double v_x = rho / rho_r * v_rx;
                v_t = -u * v_x;
            }
            else
            {
                double rho_ml = c_l / (u_m - u_l + c_l / rho_l);
                double e_ml = e_l + (pi_m - pi_l) * (pi_m + pi_l) / 2.0 / c_l / c_l;
                rho = rho_ml;
                e = e_ml;
                v = v_l;
                rho_t = -u * rho * rho * rho / (rho_l * c_l * c_l) * (c_l * c_l / rho_l / rho_l * rho_lx + c_l * u_lx + 1.5 * (u - u_l) * c_lx) - rho * rho * rho / (c_l * c_l * c_l) * (u - c_l / rho) * Dpi_Dt;
                pi_t = 1.0 / (c_l + c_r) * (c_l * c_r / rho_l * (1 + u * rho / c_r) * k_l - c_l * c_r / rho_r * (1 - u * rho / c_l) * k_r);
                u_t = 1.0 / (c_l + c_r) * (c_l / rho_l * (1 + u * rho * c_r / c_l / c_l) * k_l + c_r / rho_r * (1 - u * rho / c_l) * k_r);
                e_t = -u * rho / rho_l / c_l / c_l / c_l * (-pi_l * c_l * pi_lx + c_l * c_l * c_l * e_lx + 2.0 * (e_l - e) * c_l * c_l * c_lx) + pi / c_l / c_l * pi_t;
                double v_x = rho / rho_l * v_lx;
                v_t = -u * v_x;
            }
        }

        u_u_t_interface[2 * idx] = vec5d{rho, pi, u, v, e};
        u_u_t_interface[2 * idx + 1] = vec5d(rho_t, pi_t, u_t, v_t, e_t);
        // u_u_t_interface[2 * idx + 1] = vec5d(0, 0, 0, 0, 0);
    }
}

// convert (\rho,p,u,v) to (\rho,\rho u,\rho u^2+p,\rho v)
__device__ vec5d rhoPUVToConserVar(const vec5d &rhoPUV, double gamma)
{
    return {rhoPUV(0),
            rhoPUV(0) * rhoPUV(2),
            rhoPUV(1) / (gamma - 1.0) + rhoPUV(0) * (rhoPUV(2) * rhoPUV(2) + rhoPUV(3) * rhoPUV(3)) / 2.0,
            rhoPUV(0) * rhoPUV(3),
            0.0};
}

// convert (\rho,\rho u,\rho u^2+p,\rho v) to (\rho,p,u,v)
__device__ vec5d conserVarToRhoPUV(const vec5d &U_, double gamma)
{
    return {U_(0),
            (gamma - 1.0) * (U_(2) - U_(1) * U_(1) / U_(0) / 2.0 - U_(3) * U_(3) / U_(0) / 2.0),
            U_(1) / U_(0),
            U_(3) / U_(0),
            0.0};
}

// compute the maximum characteristic speed
__global__ void maxPropagtingSpeed(vec5d *U, double *speed, int Nx, int Ny, double gamma)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-index

    if (i < Nx && j < Ny)
    {
        int idx = j * Nx + i;
        double velo = sqrt(U[idx](2) * U[idx](2) + U[idx](3) * U[idx](3)), rho = U[idx](0), p = U[idx](1), c = sqrt(gamma * p / rho);
        speed[idx] = abs(velo) + c;
    }
}

// update using FVM method
__global__ void forward_x_dir(vec5d *U, vec5d *u_u_t_interface, int Nx, int Ny, double width, double t, double gamma)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-index
    double h = width / double(Nx);

    if (i < Nx - 2 && j < Ny && i > 1)
    {
        int idx = j * Nx + i; // cell i
        vec5d F_L = flux(u_u_t_interface[2 * idx], u_u_t_interface[2 * idx + 1], t);
        vec5d F_R = flux(u_u_t_interface[2 * idx + 2], u_u_t_interface[2 * idx + 3], t);
        vec5d U_ = rhoPUVToConserVar(U[idx], gamma);
        auto U_save = U_;
        double lambda = t / h;
        U_ = U_ - (F_R - F_L) * lambda;
        U[idx] = conserVarToRhoPUV(U_, gamma);
    }
}

// swap directions
__global__ void rotate_and_flip_vertically(vec5d *input, vec5d *output, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column in output
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row in output

    if (x < Nx && y < Ny)
    {
        int outputCol = Ny - 1 - y;
        int outputRow = Nx - 1 - x;
        int outputIdx = outputRow * Ny + outputCol;
        output[outputIdx] = input[y * Nx + x];
        swap(output[outputIdx].data[2], output[outputIdx].data[3]);
    }
}

__global__ void combineResults(vec5d *U1, vec5d *U2, int Nx, int Ny)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-index (interface)

    if (i < Nx && j < Ny)
    {
        int idx = j * Nx + i;
        U1[idx] = (U1[idx] + U2[idx]) * 0.5;
    }
}