#include <iostream>
#include "./src/wrappers.cuh"

std::vector<std::vector<vec5d>> initialize_2Mach(int N)
{
    std::vector<std::vector<vec5d>> U(N, std::vector<vec5d>(4 * N, {1.4, 1, 0, 0, 0}));
    for (auto i = N - 1; i >= 0; i--)
    {
        double temp = double(N - i - 1) / sqrt(3);
        int l = N / 6 + int(temp);
        for (auto j = 0; j < l; j++)
        {
            U[i][j] = {8, 116.5, 4.125 * sqrt(3), -4.125, 0};
        }
    }

    return U;
}

std::vector<std::vector<vec5d>> initialize_jet(int N)
{
    std::vector<std::vector<vec5d>> U(N, std::vector<vec5d>(2 * N, {0.5, 0.4127, 0.0, 0.0, 0}));
    for (auto i = 0; i < N; ++i)
    {
        if ((i >= 2 * N / 5 + 1) && (i <= 3 * N / 5))
        {
            U[i][0] = {5.0, 0.4127, 800.0, 0.0, 0};
        }
    }
    return U;
}

std::vector<std::vector<vec5d>> initialize_VSInteract(int N)
{
    double gamma = 1.4;
    std::vector<std::vector<vec5d>> U(N, std::vector<vec5d>(2 * N, {1.21, sqrt(gamma), 0, 0, 0}));
    double h = 1 / double(N);
    for (auto i = 0; i < N; i++)
    {
        for (auto j = 0; j < N / 2; j++)
        {
            double y = 1.0 - h * double(i);
            double x = h * double(j);
            double r = sqrt((x - 0.25) * (x - 0.25) + (y - 0.5) * (y - 0.5));
            double tau = r / 0.05;
            double sin_theta = abs(r) >= 1e-8 ? (y - 0.5) / r : 1;
            double cos_theta = abs(r) >= 1e-8 ? (x - 0.25) / r : 1;
            double tilde_u = 0.3 * tau * exp(0.204 * (1 - tau * tau)) * sin_theta;
            double tilde_v = -0.3 * tau * exp(0.204 * (1 - tau * tau)) * cos_theta;
            double tilde_T = -(gamma - 1.0) * 0.3 * 0.3 * exp(2.0 * 0.204 * (1 - tau * tau)) / 4.0 / 0.204 / gamma;
            double rho = pow(pow(1.21, gamma - 1) + pow(1.21, gamma) * tilde_T, 1.0 / (gamma - 1.0));
            double p = rho * (tilde_T + 1 / 1.21);
            U[i][j] = {rho, p, sqrt(gamma) + tilde_u, tilde_v, 0};
        }

        for (auto j = N / 2; j < 2 * N; j++)
        {
            double a = 1.21 * sqrt(gamma) * (gamma + 1) / 2 / (gamma - 1);
            double b = -gamma * (1 + 1.21 * gamma) / (gamma - 1);
            double c = gamma * sqrt(gamma) / (gamma - 1.0) + 1.21 * gamma * sqrt(gamma) / 2;
            double delta = sqrt(b * b - 4 * a * c);
            double u = (-b - delta) / 2 / a;
            double rho = 1.21 * sqrt(gamma) / u;
            double p = 1 + 1.21 * gamma - rho * u * u;
            U[i][j] = {rho, p, u, 0, 0};
        }
    }

    return U;
}

int main()
{
    RGRP_2D_CUDA solver(initialize_2Mach(240));
    solver.setSpatialLayout(4.0, 1.0);
    solver.setTimeLayout(0.2, 0.8);
    solver.setGhostCellStrategy(ghostCellStrategy::doubleMach);
    solver.setAlpha(2.0);
    solver.setGamma(1.4);
    solver.solve();
    solver.writeFinalResultTo_txt("./");

    std::cout << "Total " << solver.getTimeStagesCount() << " time stages with " << solver.getComputationTime_ms() << "ms" << std::endl;
    system("pause");

    solver.clear();
    return 0;
}
