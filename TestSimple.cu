#include <cuda_runtime.h>
#include "./src/basicFuns.h"
#include <chrono>
#include <fstream>

int main()
{
    // auto u_host = vec4d(0.138, 0.029, 1.206, 1.206);
    const int Nx = 4000;            // Number of cells in x-direction
    const int Ny = 4000;            // Number of cells in y-direction
    const int Nx_extended = Nx + 4; // Number of cells in x-direction
    const int Ny_extended = Ny + 4;

    double width = 1.0;
    double height = 1.0;
    double width_extended = width * double(Nx_extended) / double(Nx);
    double height_extended = height * double(Ny_extended) / double(Ny);

    int num_cells = Nx_extended * Ny_extended;

    // Host memory for cell states and fluxes
    std::vector<vec5d> u_host(num_cells);
    std::vector<vec5d> flux_x_host(num_cells);

    // Initialize cell states (example)
    for (int j = 0; j < Ny_extended; ++j)
    {
        for (int i = 0; i < Nx_extended; ++i)
        {
            if (i < Nx_extended / 2 && j < Ny_extended / 2) // 2
            {
                u_host[j * Nx_extended + i] = vec5d(0.5323, 0.3, 1.206, 0, 0);
            }
            else if (i < Nx_extended / 2 && j >= Ny_extended / 2) // 3
            {
                u_host[j * Nx_extended + i] = vec5d(0.138, 0.029, 1.206, 1.206, 0);
            }
            else if (i >= Nx_extended / 2 && j < Ny_extended / 2) // 1
            {
                u_host[j * Nx_extended + i] = vec5d(1.5, 1.5, 0, 0, 0);
            }
            else // 4
            {
                u_host[j * Nx_extended + i] = vec5d(0.5323, 0.3, 0, 1.206, 0);
            }
        }
    }

    /*for (int j = 0; j < Ny_extended; ++j)
    {
        for (int i = 0; i < Nx_extended; ++i)
        {
            if (i < Nx_extended / 2 && j < Ny_extended / 2) // 2
            {
                u_host[j * Nx_extended + i] = vec5d(2.0, 1.0, 0.75, 0.5, 0);
            }
            else if (i < Nx_extended / 2 && j >= Ny_extended / 2) // 3
            {
                u_host[j * Nx_extended + i] = vec5d(1.0, 1.0, -0.75, 0.5, 0);
            }
            else if (i >= Nx_extended / 2 && j < Ny_extended / 2) // 1
            {
                u_host[j * Nx_extended + i] = vec5d(1, 1, 0.75, -0.5, 0);
            }
            else // 4
            {
                u_host[j * Nx_extended + i] = vec5d(3, 1.0, -0.75, -0.5, 0);
            }
        }
    }*/


    /*for (int j = 0; j < Ny_extended; ++j)
    {
        for (int i = 0; i < Nx_extended; ++i)
        {
            if (i < Nx_extended / 2 && j < Ny_extended / 2) // 2
            {
                u_host[j * Nx_extended + i] = vec5d(1, 1, -0.6259, 0.1, 0);
            }
            else if (i < Nx_extended / 2 && j >= Ny_extended / 2) // 3
            {
                u_host[j * Nx_extended + i] = vec5d(0.8, 1, 0.1, 0.1, 0);
            }
            else if (i >= Nx_extended / 2 && j < Ny_extended / 2) // 1
            {
                u_host[j * Nx_extended + i] = vec5d(0.5197, 0.4, 0.1, 0.1, 0);
            }
            else // 4
            {
                u_host[j * Nx_extended + i] = vec5d(1, 1, 0.1, -0.6259, 0);
            }
        }
    }*/


    /*for (int j = 0; j < Ny_extended; ++j)
    {
        for (int i = 0; i < Nx_extended; ++i)
        {
            if (i < Nx_extended / 2 && j < Ny_extended / 2) // 2
            {
                u_host[j * Nx_extended + i] = vec5d(0.5197, 0.4, -0.6259, 0.1, 0);
            }
            else if (i < Nx_extended / 2 && j >= Ny_extended / 2) // 3
            {
                u_host[j * Nx_extended + i] = vec5d(0.8, 0.4, 0.1, 0.1, 0);
            }
            else if (i >= Nx_extended / 2 && j < Ny_extended / 2) // 1
            {
                u_host[j * Nx_extended + i] = vec5d(1, 1, 0.1, 0.1, 0);
            }
            else // 4
            {
                u_host[j * Nx_extended + i] = vec5d(0.5197, 0.4, 0.1, -0.6259, 0);
            }
        }
    }*/

    // jet
    /*for (int j = 0; j < Ny_extended; ++j)
    {
        for (auto i = 0; i < Nx_extended; i++)
        {
            if (i == 0 && j >= 2 * Ny_extended / 5 + 1 && j <= 3 * Ny_extended / 5)
            {
                u_host[j * Nx_extended + i] = {5.0, 0.4127, 800.0, 0.0, 0.0};
            }
            else
            {
                u_host[j * Nx_extended + i] = {0.5, 0.4127, 0.0, 0.0, 0.0};
            }
        }
    }*/

    // 2mach
    /*for (int j = 0; j < Ny_extended; ++j)
    {
        double temp = double(Ny_extended - j) / sqrt(3);
        int l = Ny_extended / 6 + int(temp)+1;//+2+3+1
        for (auto i = 0; i < l; i++)
        {
            u_host[j * Nx_extended + i] = {8, 116.5, 4.125 * sqrt(3), -4.125, 0};
        }
        for (auto i = l; i < Nx_extended; i++)
        {
            u_host[j * Nx_extended + i] = {1.4, 1, 0, 0, 0};
        }
    }*/

    /*for (int j = 0; j < Ny_extended; ++j)
    {
        double gamma = 1.4;
        for (auto i = 0; i < Nx_extended; i++)
        {
            u_host[j * Nx_extended + i] = {1.21, sqrt(gamma), 0, 0, 0};
        }
        double h = 1 / double(Ny_extended);
        for (auto i = 0; i < Ny_extended / 2; i++)
        {
            double y = 1.0 - h * double(j);
            double x = h * double(i);
            double r = sqrt((x - 0.25) * (x - 0.25) + (y - 0.5) * (y - 0.5));
            double tau = r / 0.05;
            double sin_theta = abs(r) >= 1e-8 ? (y - 0.5) / r : 1;
            double cos_theta = abs(r) >= 1e-8 ? (x - 0.25) / r : 1;
            double tilde_u = 0.3 * tau * exp(0.204 * (1 - tau * tau)) * sin_theta;
            double tilde_v = -0.3 * tau * exp(0.204 * (1 - tau * tau)) * cos_theta;
            double tilde_T = -(gamma - 1.0) * 0.3 * 0.3 * exp(2.0 * 0.204 * (1 - tau * tau)) / 4.0 / 0.204 / gamma;
            double rho = pow(pow(1.21, gamma - 1) + pow(1.21, gamma) * tilde_T, 1.0 / (gamma - 1.0));
            double p = rho * (tilde_T + 1 / 1.21);
            u_host[j * Nx_extended + i] = {rho, p, sqrt(gamma) + tilde_u, tilde_v, 0.0};
        }

        for (auto i = Ny_extended / 2; i < 2 * Ny_extended; i++)
        {
            double a = 1.21 * sqrt(gamma) * (gamma + 1) / 2 / (gamma - 1);
            double b = -gamma * (1 + 1.21 * gamma) / (gamma - 1);
            double c = gamma * sqrt(gamma) / (gamma - 1.0) + 1.21 * gamma * sqrt(gamma) / 2;
            double delta = sqrt(b * b - 4 * a * c);
            double u = (-b - delta) / 2 / a;
            double rho = 1.21 * sqrt(gamma) / u;
            double p = 1 + 1.21 * gamma - rho * u * u;
            u_host[j * Nx_extended + i] = {rho, p, u, 0, 0};
        }
    }*/

    // Allocate device memory
    vec5d *u_dev;
    vec5d *u_dev_copy;
    // vec5d *u_dev_y_copy;
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
    // double *c_lr_interface_dev;
    // double *c_lr_y_interface_dev;
    cudaMalloc(&u_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_u_t_interface_dev, (num_cells + Ny_extended) * 2 * sizeof(vec5d));
    cudaMalloc(&u_lr_interface_dev, (num_cells + Ny_extended) * 2 * sizeof(vec5d));
    cudaMalloc(&u_u_t_y_interface_dev, (num_cells + Nx_extended) * 2 * sizeof(vec5d));
    cudaMalloc(&u_y_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_dev_copy, num_cells * sizeof(vec5d));
    // cudaMalloc(&u_dev_y_copy, num_cells * sizeof(vec5d));
    cudaMalloc(&c_lr_interface_edited_dev, (num_cells + Ny_extended) * 2 * sizeof(int));
    cudaMalloc(&c_lr_interface_y_edited_dev, (num_cells + Ny_extended) * 2 * sizeof(int));
    cudaMalloc(&u_lr_y_interface_dev, (num_cells + Nx_extended) * 2 * sizeof(vec5d));
    cudaMalloc(&u_slope_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&u_slope_y_dev, num_cells * sizeof(vec5d));
    cudaMalloc(&speed, num_cells * sizeof(double));
    // cudaMalloc(&c_lr_interface_dev, (num_cells + Ny) * 2 * sizeof(double));
    // cudaMalloc(&c_lr_y_interface_dev, (num_cells + Nx) * 2 * sizeof(vec5d));
    cudaMemset(u_slope_dev, 0, num_cells * sizeof(vec5d));
    cudaMemset(c_lr_interface_edited_dev, 0, (num_cells + Ny_extended) * 2 * sizeof(int));
    cudaMemset(c_lr_interface_y_edited_dev, 0, (num_cells + Ny_extended) * 2 * sizeof(int));

    // Copy data to device
    cudaMemcpy(u_dev, u_host.data(), num_cells * sizeof(vec5d), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((Nx_extended + blockSize.x - 1) / blockSize.x, (Ny_extended + blockSize.y - 1) / blockSize.y);
    dim3 gridSize_y((Ny_extended + blockSize.y - 1) / blockSize.y, (Nx_extended + blockSize.x - 1) / blockSize.x);

    double cTime = 0.0;
    double alpha = 1.0;
    double CFL = 0.5;
    double endT = 0.35;
    double gamma = 1.4;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();
    while (cTime < endT)
    {
        maxPropagtingSpeed<<<gridSize, blockSize>>>(u_dev, speed, Nx_extended, Ny_extended, gamma);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> p_speed = thrust::device_pointer_cast(speed);
        double max_speed = reduce(p_speed, p_speed + num_cells, -std::numeric_limits<double>::infinity(), thrust::maximum<double>());
        double h = std::min(width_extended / double(Nx_extended), height_extended / double(Ny_extended));
        double dt = std::min(CFL * h / max_speed, endT - cTime);
        if (h / max_speed < 1e-20)
        {
            break;
        }

        cudaMemcpy(u_dev_copy, u_dev, num_cells * sizeof(vec5d), cudaMemcpyDeviceToDevice);

        MUSCL_rel_RP_GRP<<<gridSize, blockSize>>>(u_dev, u_slope_dev, Nx_extended, Ny_extended, alpha, width_extended, u_lr_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize, blockSize>>>(u_dev, u_slope_dev, u_lr_interface_dev, Nx_extended, Ny_extended, u_u_t_interface_dev, c_lr_interface_edited_dev, gamma);
        cudaDeviceSynchronize();
        time_deris<<<gridSize, blockSize>>>(u_u_t_interface_dev, u_lr_interface_dev, u_slope_dev, Nx_extended, Ny_extended, width_extended, c_lr_interface_edited_dev, gamma);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize, blockSize>>>(u_dev, u_u_t_interface_dev, Nx_extended, Ny_extended, width_extended, dt, gamma);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize, blockSize>>>(u_dev, u_y_dev, Nx_extended, Ny_extended);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, Ny_extended, Nx_extended, alpha, height_extended, u_lr_y_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, u_lr_y_interface_dev, Ny_extended, Nx_extended, u_u_t_y_interface_dev, c_lr_interface_y_edited_dev, gamma);
        cudaDeviceSynchronize();
        time_deris<<<gridSize_y, blockSize>>>(u_u_t_y_interface_dev, u_lr_y_interface_dev, u_slope_y_dev, Ny_extended, Nx_extended, height_extended, c_lr_interface_y_edited_dev, gamma);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize_y, blockSize>>>(u_y_dev, u_u_t_y_interface_dev, Ny_extended, Nx_extended, height_extended, dt, gamma);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize_y, blockSize>>>(u_y_dev, u_dev, Ny_extended, Nx_extended);
        cudaDeviceSynchronize();

        rotate_and_flip_vertically<<<gridSize, blockSize>>>(u_dev_copy, u_y_dev, Nx_extended, Ny_extended);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, Ny_extended, Nx_extended, alpha, height_extended, u_lr_y_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize_y, blockSize>>>(u_y_dev, u_slope_y_dev, u_lr_y_interface_dev, Ny_extended, Nx_extended, u_u_t_y_interface_dev, c_lr_interface_y_edited_dev, gamma);
        cudaDeviceSynchronize();
        time_deris<<<gridSize_y, blockSize>>>(u_u_t_y_interface_dev, u_lr_y_interface_dev, u_slope_y_dev, Ny_extended, Nx_extended, height_extended, c_lr_interface_y_edited_dev, gamma);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize_y, blockSize>>>(u_y_dev, u_u_t_y_interface_dev, Ny_extended, Nx_extended, height_extended, dt, gamma);
        cudaDeviceSynchronize();
        rotate_and_flip_vertically<<<gridSize_y, blockSize>>>(u_y_dev, u_dev_copy, Ny_extended, Nx_extended);
        cudaDeviceSynchronize();
        MUSCL_rel_RP_GRP<<<gridSize, blockSize>>>(u_dev_copy, u_slope_dev, Nx_extended, Ny_extended, alpha, width_extended, u_lr_interface_dev);
        cudaDeviceSynchronize();
        rel_RP_posi_fix<<<gridSize, blockSize>>>(u_dev_copy, u_slope_dev, u_lr_interface_dev, Nx_extended, Ny_extended, u_u_t_interface_dev, c_lr_interface_edited_dev, gamma);
        cudaDeviceSynchronize();
        time_deris<<<gridSize, blockSize>>>(u_u_t_interface_dev, u_lr_interface_dev, u_slope_dev, Nx_extended, Ny_extended, width_extended, c_lr_interface_edited_dev, gamma);
        cudaDeviceSynchronize();
        forward_x_dir<<<gridSize, blockSize>>>(u_dev_copy, u_u_t_interface_dev, Nx_extended, Ny_extended, width_extended, dt, gamma);
        cudaDeviceSynchronize();

        combineResults<<<gridSize, blockSize>>>(u_dev, u_dev_copy, Nx_extended, Ny_extended);

        cTime += dt;

        set_ghost_cells<<<gridSize, blockSize>>>(u_dev, Nx_extended, Ny_extended, width_extended, height_extended, cTime);
        cudaDeviceSynchronize();
        cudaMemset(c_lr_interface_edited_dev, 0, (num_cells + Ny_extended) * 2 * sizeof(int));
        cudaMemset(c_lr_interface_y_edited_dev, 0, (num_cells + Ny_extended) * 2 * sizeof(int));
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy fluxes back to host
    cudaMemcpy(u_host.data(), u_dev, num_cells * sizeof(vec5d), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(u_dev);

    std::ofstream file_rho("test_rho200_test.txt");
    std::ofstream file_p("test_p200.txt");
    std::ofstream file_u("test_u200.txt");
    if (file_rho.is_open())
    {
        for (int j = 0; j < Ny_extended; ++j)
        {
            for (int i = 0; i < Nx_extended; ++i)
            {
                file_rho << u_host[j * Nx_extended + i](0) << " ";
                file_p << u_host[j * Nx_extended + i](1) << " ";
            }
            file_rho << "\n";
            file_p << "\n";
            file_u << "\n";
        }
    }
    file_rho.close();
    file_p.close();
    file_u.close();

    std::cout << duration.count() / 1000.0 << "ms";
    system("pause");
    return 0;
}
