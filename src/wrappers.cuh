#pragma once

#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "basicFuns.cuh"
using std::ofstream;
using std::string;
using std::vector;
namespace fs = std::filesystem;

enum class ghostCellStrategy
{
    outflow,
    reflective,
    jet,
    jet800,
    doubleMach
};

class RGRP_2D_CUDA
{
public:
    RGRP_2D_CUDA(const vector<vector<vec5d>> &U0);
    RGRP_2D_CUDA() {};
    void setRPInitialData(const vec5d &U1, const vec5d &U2, const vec5d &U3, const vec5d &U4, int Nx, int Ny);
    void setSpatialLayout(double width, double height);
    void setTimeLayout(double endTime, double CFL);
    void setTimeLayout(double endTime);
    void setCFL(double CFL);
    void setAlpha(double alpha);
    void setGamma(double gamma);
    void setGhostCellStrategy(ghostCellStrategy strategy);
    void setRecordModel(bool on_off, int gapPerFrame = 10, const string &recordDir = ".");
    vector<vector<vec5d>> solve();
    double getComputationTime_ms();
    int getTimeStagesCount();
    void writeFinalResultTo_txt(const string &dir);
    void clear();

protected:
    vector<vec5d> __U0__;
    vector<vector<vec5d>> __U__;
    double __width__ = 1.0, __height__ = 1.0, __endTime__ = 0.0, __CFL__ = 0.5;
    double __alpha__ = 1.9, __gamma__ = 1.4;
    int __Nx__ = 100, __Ny__ = 100;
    ghostCellStrategy __strategy__ = ghostCellStrategy::outflow;

    bool __recordAllTimeStages__ = false;
    vector<vector<vector<vec5d>>> __allTimeStages__;
    vector<double> __cTimes__;
    double __computationTime_ms__ = 0.0;
    int __timeStagesNum__ = 0;
    int __gapPerFrame__ = 1;
    string __recordDir__ = ".";

    vector<vector<vec5d>> flattened2vectorized(const vector<vec5d> &flattened);
    static void writeMatrixTo_txt(const string &dir, const vector<vector<vec5d>> &matrix, double cTime);
    void writeStagesTo_png(const string &dir, const vector<vector<vec5d>> &data, int i);
};

RGRP_2D_CUDA::RGRP_2D_CUDA(const vector<vector<vec5d>> &U0)
{
    for (auto &row : U0)
    {
        for (auto &u : row)
        {
            __U0__.push_back(u);
        }
    }
    __Nx__ = U0[0].size();
    __Ny__ = U0.size();
}

void RGRP_2D_CUDA::setRPInitialData(const vec5d &U1, const vec5d &U2, const vec5d &U3, const vec5d &U4, int Nx, int Ny)
{
    __U0__.clear();
    __U0__.resize(Nx * Ny);
    __Nx__ = Nx;
    __Ny__ = Ny;
    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            if (i < Nx / 2 && j < Ny / 2) // 2
            {
                __U0__[j * Nx + i] = U2;
            }
            else if (i < Nx / 2 && j >= Ny / 2) // 3
            {
                __U0__[j * Nx + i] = U3;
            }
            else if (i >= Nx / 2 && j < Ny / 2) // 1
            {
                __U0__[j * Nx + i] = U1;
            }
            else // 4
            {
                __U0__[j * Nx + i] = U4;
            }
        }
    }
}

void RGRP_2D_CUDA::setSpatialLayout(double width, double height)
{
    __width__ = width;
    __height__ = height;
}

void RGRP_2D_CUDA::setTimeLayout(double endTime, double CFL)
{
    __endTime__ = endTime;
    __CFL__ = CFL;
}

void RGRP_2D_CUDA::setTimeLayout(double endTime)
{
    __endTime__ = endTime;
}

void RGRP_2D_CUDA::setCFL(double CFL)
{
    __CFL__ = CFL;
}

void RGRP_2D_CUDA::setAlpha(double alpha)
{
    __alpha__ = alpha;
}

void RGRP_2D_CUDA::setGamma(double gamma)
{
    __gamma__ = gamma;
}

void RGRP_2D_CUDA::setGhostCellStrategy(ghostCellStrategy strategy)
{
    __strategy__ = strategy;
}

void RGRP_2D_CUDA::setRecordModel(bool on_off, int gapPerFrame, const string &recordDir)
{
    __recordAllTimeStages__ = on_off;
    __gapPerFrame__ = gapPerFrame;
    __recordDir__ = recordDir;
}

vector<vector<vec5d>> RGRP_2D_CUDA::solve()
{
    vec5d *u_dev;
    vec5d *u_dev_copy;
    vec5d *u_u_t_interface_dev;
    vec5d *u_lr_interface_dev;
    vec5d *u_y_dev;
    vec5d *u_u_t_y_interface_dev;
    vec5d *u_lr_y_interface_dev;
    vec5d *u_slope_dev;
    vec5d *u_slope_y_dev;
    double *speed;
    int *c_lr_interface_edited_dev;
    int *c_lr_interface_y_edited_dev;

    int num_cells = __Nx__ * __Ny__;
    cudaMalloc(&u_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_u_t_interface_dev, (num_cells + __Ny__) * 2 * sizeof(vec5d));
    cudaMalloc(&u_lr_interface_dev, (num_cells + __Ny__) * 2 * sizeof(vec5d));
    cudaMalloc(&u_u_t_y_interface_dev, (num_cells + __Nx__) * 2 * sizeof(vec5d));
    cudaMalloc(&u_y_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_dev_copy, num_cells * sizeof(vec5d));
    cudaMalloc(&c_lr_interface_edited_dev, (num_cells + __Ny__) * 2 * sizeof(int));
    cudaMalloc(&c_lr_interface_y_edited_dev, (num_cells + __Ny__) * 2 * sizeof(int));
    cudaMalloc(&u_lr_y_interface_dev, (num_cells + __Nx__) * 2 * sizeof(vec5d));
    cudaMalloc(&u_slope_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_slope_y_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&speed, num_cells * sizeof(double));
    cudaMemset(u_slope_dev, 0, num_cells * sizeof(vec5d));
    cudaMemset(c_lr_interface_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));
    cudaMemset(c_lr_interface_y_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));
    cudaMemcpy(u_dev, __U0__.data(), num_cells * sizeof(vec5d), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((__Nx__ + blockSize.x - 1) / blockSize.x, (__Ny__ + blockSize.y - 1) / blockSize.y);
    dim3 gridSize_y((__Ny__ + blockSize.y - 1) / blockSize.y, (__Nx__ + blockSize.x - 1) / blockSize.x);

    double cTime = 0.0;
    __cTimes__.push_back(0.0);
    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;
    int iter_ = 0;
    while (cTime < __endTime__)
    {
        maxPropagtingSpeed<<<gridSize, blockSize>>>(u_dev, speed, __Nx__, __Ny__, __gamma__);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> p_speed = thrust::device_pointer_cast(speed);
        double max_speed = reduce(p_speed, p_speed + num_cells, -std::numeric_limits<double>::infinity(), thrust::maximum<double>());
        double h = std::min(__width__ / double(__Nx__), __height__ / double(__Ny__));
        double dt = std::min(__CFL__ * h / max_speed, __endTime__ - cTime);
        if (h / max_speed < 1e-20)
        {
            break;
        }

        cudaMemcpy(u_dev_copy, u_dev, num_cells * sizeof(vec5d), cudaMemcpyDeviceToDevice);

        MUSCL_rel_RP_GRP<<<gridSize, blockSize>>>(u_dev, u_slope_dev, __Nx__, __Ny__, __alpha__, __width__, u_lr_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize, blockSize>>>(u_dev, u_slope_dev, u_lr_interface_dev, __Nx__, __Ny__, u_u_t_interface_dev, c_lr_interface_edited_dev, __gamma__);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize, blockSize>>>(u_dev, u_u_t_interface_dev, __Nx__, __Ny__, __width__, dt, __gamma__);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize, blockSize>>>(u_dev, u_y_dev, __Nx__, __Ny__);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, __Ny__, __Nx__, __alpha__, __height__, u_lr_y_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, u_lr_y_interface_dev, __Ny__, __Nx__, u_u_t_y_interface_dev, c_lr_interface_y_edited_dev, __gamma__);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize_y, blockSize>>>(u_y_dev, u_u_t_y_interface_dev, __Ny__, __Nx__, __height__, dt, __gamma__);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize_y, blockSize>>>(u_y_dev, u_dev, __Ny__, __Nx__);
        cudaDeviceSynchronize();

        cudaMemset(c_lr_interface_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));
        cudaMemset(c_lr_interface_y_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));

        rotate_and_flip_vertically<<<gridSize, blockSize>>>(u_dev_copy, u_y_dev, __Nx__, __Ny__);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, __Ny__, __Nx__, __alpha__, __height__, u_lr_y_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, u_lr_y_interface_dev, __Ny__, __Nx__, u_u_t_y_interface_dev, c_lr_interface_y_edited_dev, __gamma__);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize_y, blockSize>>>(u_y_dev, u_u_t_y_interface_dev, __Ny__, __Nx__, __height__, dt, __gamma__);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize_y, blockSize>>>(u_y_dev, u_dev_copy, __Ny__, __Nx__);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize, blockSize>>>(u_dev_copy, u_slope_dev, __Nx__, __Ny__, __alpha__, __width__, u_lr_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize, blockSize>>>(u_dev_copy, u_slope_dev, u_lr_interface_dev, __Nx__, __Ny__, u_u_t_interface_dev, c_lr_interface_edited_dev, __gamma__);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize, blockSize>>>(u_dev_copy, u_u_t_interface_dev, __Nx__, __Ny__, __width__, dt, __gamma__);
        cudaDeviceSynchronize();

        combineResults<<<gridSize, blockSize>>>(u_dev, u_dev_copy, __Nx__, __Ny__);

        cTime += dt;
        __cTimes__.push_back(cTime);
        __timeStagesNum__++;
        if (__recordAllTimeStages__)
        {
            if (iter == 0 || iter == __gapPerFrame__)
            {
                cudaMemcpy(__U0__.data(), u_dev, num_cells * sizeof(vec5d), cudaMemcpyDeviceToHost);
                writeStagesTo_png(__recordDir__, flattened2vectorized(__U0__), iter_);
                iter = 0;
            }
        }
        iter++;
        iter_++;

        if (__strategy__ == ghostCellStrategy::outflow)
        {
            set_ghost_cells<<<gridSize, blockSize>>>(u_dev, __Nx__, __Ny__, __width__, __height__, cTime);
        }
        else if (__strategy__ == ghostCellStrategy::jet)
        {
            set_ghost_cells_jet<<<gridSize, blockSize>>>(u_dev, __Nx__, __Ny__, __width__, __height__, cTime);
        }
        else if (__strategy__ == ghostCellStrategy::jet800)
        {
            set_ghost_cells_jet800<<<gridSize, blockSize>>>(u_dev, __Nx__, __Ny__, __width__, __height__, cTime);
        }
        else
        {
            set_ghost_cells_2Mach<<<gridSize, blockSize>>>(u_dev, __Nx__, __Ny__, __width__, __height__, cTime);
        }
        cudaDeviceSynchronize();
        cudaMemset(c_lr_interface_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));
        cudaMemset(c_lr_interface_y_edited_dev, 0, (num_cells + __Ny__) * 2 * sizeof(int));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    __computationTime_ms__ = duration.count() / 1000.0;

    cudaMemcpy(__U0__.data(), u_dev, num_cells * sizeof(vec5d), cudaMemcpyDeviceToHost);
    __U__ = flattened2vectorized(__U0__);
    cudaFree(u_dev);
    cudaFree(u_u_t_interface_dev);
    cudaFree(u_lr_interface_dev);
    cudaFree(u_u_t_y_interface_dev);
    cudaFree(u_y_dev);
    cudaFree(u_dev_copy);
    cudaFree(c_lr_interface_edited_dev);
    cudaFree(c_lr_interface_y_edited_dev);
    cudaFree(u_lr_y_interface_dev);
    cudaFree(u_slope_dev);
    cudaFree(u_slope_y_dev);
    cudaFree(speed);

    return __U__;
}

double RGRP_2D_CUDA::getComputationTime_ms()
{
    return __computationTime_ms__;
}

int RGRP_2D_CUDA::getTimeStagesCount()
{
    return __timeStagesNum__;
}

void RGRP_2D_CUDA::writeFinalResultTo_txt(const string &dir)
{
    writeMatrixTo_txt(dir, __U__, __endTime__);
}

void RGRP_2D_CUDA::writeStagesTo_png(const string &dir, const vector<vector<vec5d>> &data, int i)
{
    int colormap = cv::COLORMAP_VIRIDIS;
    bool normalize = false;
    cv::Mat mat(data.size(), data[0].size(), CV_32FC1);
    for (size_t row = 0; row < data.size(); ++row)
    {
        for (size_t col = 0; col < data[row].size(); ++col)
        {
            mat.at<float>(row, col) = data[row][col](0);
        }
    }
    CV_Assert(mat.type() == CV_32FC1 || mat.type() == CV_64FC1);

    cv::Mat display_mat;
    cv::normalize(mat, display_mat, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Apply colormap
    fs::create_directories(dir);
    cv::Mat colored;
    cv::applyColorMap(display_mat, colored, colormap);
    cv::imwrite(dir + "/" + std::to_string(i) + ".png", colored);
}

void RGRP_2D_CUDA::clear()
{
    __U0__.clear();
    __U__.clear();
    __allTimeStages__.clear();
    __cTimes__.clear();
}

vector<vector<vec5d>> RGRP_2D_CUDA::flattened2vectorized(const vector<vec5d> &flattened)
{
    vector<vector<vec5d>> ret;
    for (int j = 0; j < __Ny__; ++j)
    {
        vector<vec5d> temp;
        for (int i = 0; i < __Nx__; ++i)
        {
            temp.push_back(flattened[j * __Nx__ + i]);
        }
        ret.push_back(temp);
    }
    return ret;
}

void RGRP_2D_CUDA::writeMatrixTo_txt(const string &dir, const vector<vector<vec5d>> &matrix, double cTime)
{
    ofstream file_rho(dir + "/rho_" + std::to_string(cTime) + "s.txt");
    ofstream file_p(dir + "/p_" + std::to_string(cTime) + "s.txt");
    ofstream file_u(dir + "/u_" + std::to_string(cTime) + "s.txt");
    ofstream file_v(dir + "/v_" + std::to_string(cTime) + "s.txt");
    if (file_rho.is_open())
    {
        for (const auto &row : matrix)
        {
            for (const auto &elem : row)
            {
                file_rho << elem(0) << " ";
                file_p << elem(1) << " ";
                file_u << elem(2) << " ";
                file_v << elem(3) << " ";
            }
            file_rho << "\n";
            file_p << "\n";
            file_u << "\n";
            file_v << "\n";
        }
    }
    file_rho.close();
    file_p.close();
    file_u.close();
    file_v.close();
}
