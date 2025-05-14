#include <iostream>
#include "./src/wrappers.cuh"

std::vector<std::vector<vec5d>> initialize_2Mach(int N)
{
    std::vector<std::vector<vec5d>> U(N, std::vector<vec5d>(4 * N, {1.4, 1, 0, 0, 0}));
    for (auto i = N - 1; i >= 0; i--)
    {
        double temp = double(N - i + 3) / sqrt(3);
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

std::vector<std::vector<vec5d>> initialize_jet800(int N)
{
    std::vector<std::vector<vec5d>> U(N, std::vector<vec5d>(3 * N, {0.14, 1, 0, 0, 0}));
    for (auto i = 0; i < N; ++i)
    {
        if (i > 9 * N / 10)
        {
            U[i][0] = {1.4, 1, 800, 0, 0};
        }
    }
    return U;
}

std::vector<std::vector<vec5d>> initialize_4shocks(int N)
{
    vec5d U1 = {1.5, 1.5, 0.0, 0.0, 0};
    vec5d U2 = {0.5323, 0.3, 1.206, 0.0, 0};
    vec5d U3 = {0.138, 0.029, 1.206, 1.206, 0};
    vec5d U4 = {0.5323, 0.3, 0.0, 1.206, 0};
    vector<vector<vec5d>> ret;
    for (auto i = 0; i < N / 5; i++)
    {
        vector<vec5d> x_oriented_i(N);
        for (auto j = 0; j < 4 * N / 5; j++)
        {
            x_oriented_i[j] = U2;
        }
        for (auto j = 4 * N / 5; j < N; j++)
        {
            x_oriented_i[j] = U1;
        }
        ret.push_back(x_oriented_i);
    }
    for (auto i = N / 5; i < N; i++)
    {
        vector<vec5d> x_oriented_i(N);
        for (auto j = 0; j < 4 * N / 5; j++)
        {
            x_oriented_i[j] = U3;
        }
        for (auto j = 4 * N / 5; j < N; j++)
        {
            x_oriented_i[j] = U4;
        }
        ret.push_back(x_oriented_i);
    }
    return ret;
}

std::vector<std::vector<vec5d>> initialize_4cds(int N)
{
    vec5d U1 = {1.0, 1.0, 0.75, -0.5, 0};
    vec5d U2 = {2.0, 1.0, 0.75, 0.5, 0};
    vec5d U3 = {1.0, 1.0, -0.75, 0.5, 0};
    vec5d U4 = {3.0, 1.0, -0.75, -0.5, 0};
    vector<vector<vec5d>> ret;
    for (auto i = 0; i < N / 2; i++)
    {
        vector<vec5d> x_oriented_i(N);
        for (auto j = 0; j < N / 2; j++)
        {
            x_oriented_i[j] = U2;
        }
        for (auto j = N / 2; j < N; j++)
        {
            x_oriented_i[j] = U1;
        }
        ret.push_back(x_oriented_i);
    }
    for (auto i = N / 2; i < N; i++)
    {
        vector<vec5d> x_oriented_i(N);
        for (auto j = 0; j < N / 2; j++)
        {
            x_oriented_i[j] = U3;
        }
        for (auto j = N / 2; j < N; j++)
        {
            x_oriented_i[j] = U4;
        }
        ret.push_back(x_oriented_i);
    }
    return ret;
}

std::vector<std::vector<vec5d>> initialize_Sedov1D(int N)
{
    const double E0 = 3200000.0;
    const double domain_size = 4.0;
    const double dx = 2 * domain_size / N;

    vec5d centerValue(1.0, 0.4 * E0 / dx, 0.0, 0, 0);
    vec5d defaultValue(1.0, 0.4 * 1e-12, 0.0, 0, 0);
    vector<vec5d> U(N, defaultValue);

    U[N / 2] = centerValue;
    U[N / 2 - 1] = centerValue;

    std::vector<std::vector<vec5d>> ret(N, U);
    return ret;
}

int main()
{
    RGRP_2D_CUDA solver(initialize_4shocks(2000));
    solver.setSpatialLayout(1.0, 1.0);
    solver.setTimeLayout(0.8, 0.5);
    solver.setGhostCellStrategy(ghostCellStrategy::outflow);
    solver.setAlpha(2.0);
    solver.setGamma(1.4);
    solver.setRecordModel(true, 18, "./pngs"); //comment out this line if you don't want to output pngs and create videos; this line significantly reduces the efficiency.
    solver.solve();
    solver.writeFinalResultTo_txt(".");

    std::cout << "Total " << solver.getTimeStagesCount() << " time stages with " << solver.getComputationTime_ms() << "ms" << std::endl;
    system("pause");

    solver.clear();
    return 0;
}
