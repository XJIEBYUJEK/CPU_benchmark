#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>



inline uint64_t rdtsc() {
    uint32_t lo, hi;
    asm volatile ("rdtsc\n" : "=a" (lo), "=d" (hi));
    return ((uint64_t) hi << 32) | lo;
}

inline void benchmark(const std::vector<double> &first, const std::vector<double> &second) {
    uint64_t operations = first.size();

    double var1;
    double var2;
    double var3;

    uint64_t processor_clocks_dependent = 0, processor_clocks_independent = 0;
    double time_dependent = 0, time_independent = 0;
    std::chrono::duration<double> duration{};
    auto start_time = std::chrono::system_clock::now();

    for (uint64_t i = 0; i < operations; ++i) {
        var1= first[i];
        var2 = second[i];

        start_time = std::chrono::system_clock::now();
        uint64_t start = rdtsc();
        var1= var1 * var2;
        uint64_t end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_dependent += duration.count();
        processor_clocks_dependent += end - start;

        start_time = std::chrono::system_clock::now();
        start = rdtsc();
        var3 = var1 * var2;
        end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_independent += duration.count();
        processor_clocks_independent += end - start;
    }
    std::cout << "Default:" << '\n'
            << "Латентность: \n"
            << " Время: " << time_dependent << " с\n"
            << " Такты: " << processor_clocks_dependent << "\n"
            << " Тактов на одну операцию: " << (double) processor_clocks_dependent / operations << "\n"
            << "Темп выдачи результатов: \n"
            << " Время: " << time_independent << " с\n"
            << " Такты: " << processor_clocks_independent << "\n"
            << " Тактов на одну операцию: " << (double) processor_clocks_independent / operations << "\n\n";
}

inline void sse2_benchmark(const std::vector<double> &first, const std::vector<double> &second) {
    uint64_t operations = first.size() / 2;

    __m128d var1;
    __m128d var2;
    __m128d var3;

    uint64_t processor_clocks_dependent = 0, processor_clocks_independent = 0;
    double time_dependent = 0, time_independent = 0;
    std::chrono::duration<double> duration{};
    auto start_time = std::chrono::system_clock::now();

    for (uint64_t i = 0; i < operations; i += 2) {
        var1 = _mm_loadu_pd(&first[i]);
        var2 = _mm_loadu_pd(&second[i]);

        start_time = std::chrono::system_clock::now();
        uint64_t start = rdtsc();
        var1 = _mm_mul_pd(var1, var2);
        uint64_t end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_dependent += duration.count();
        processor_clocks_dependent += end - start;

        start_time = std::chrono::system_clock::now();
        start = rdtsc();
        var3 = _mm_mul_pd(var1, var2);
        end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_independent += duration.count();
        processor_clocks_independent += end - start;
    }
    processor_clocks_dependent *= 2;
    processor_clocks_independent *= 2;
    time_dependent *= 2;
    time_independent *= 2;
    std::cout << "SSE2:" << '\n'
            << "Латентность: \n"
                << " Время: " << time_dependent << " с\n"
                << " Такты: " << processor_clocks_dependent << "\n"
                << " Тактов на одну операцию: " << (double) processor_clocks_dependent / operations << "\n"
            << "Темп выдачи результатов: \n"
                << " Время: " << time_independent << " с\n"
                << " Такты: " << processor_clocks_independent << "\n"
                << " Тактов на одну операцию: " << (double) processor_clocks_independent / operations << "\n\n";
}

inline void avx2_benchmark(const std::vector<double> &first, const std::vector<double> &second) {
    uint64_t operations = first.size() / 4;

    __m256d var1;
    __m256d var2;
    __m256d var3;

    uint64_t processor_clocks_dependent = 0, processor_clocks_independent = 0;
    double time_dependent = 0, time_independent = 0;
    std::chrono::duration<double> duration{};
    auto start_time = std::chrono::system_clock::now();

    for (uint64_t i = 0; i < operations; i += 4) {
        var1 = _mm256_loadu_pd(&first[i]);
        var2 = _mm256_loadu_pd(&second[i]);

        start_time = std::chrono::system_clock::now();
        uint64_t start = rdtsc();
        var1 = _mm256_mul_pd(var1, var2);
        uint64_t end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_dependent += duration.count();
        processor_clocks_dependent += end - start;

        start_time = std::chrono::system_clock::now();
        start = rdtsc();
        var3 = _mm256_mul_pd(var1, var2);
        end = rdtsc();
        duration = std::chrono::system_clock::now() - start_time;
        time_independent += duration.count();
        processor_clocks_independent += end - start;
    }
    processor_clocks_dependent *= 4;
    processor_clocks_independent *= 4;
    time_dependent *= 4;
    time_independent *= 4;
    std::cout << "AVX2:" << '\n'
            << "Латентность: \n"
            << " Время: " << time_dependent << " с\n"
            << " Такты: " << processor_clocks_dependent << "\n"
            << " Тактов на одну операцию: " << (double) processor_clocks_dependent / operations << "\n"
            << "Темп выдачи результатов: \n"
            << " Время: " << time_independent << " с\n"
            << " Такты: " << processor_clocks_independent << "\n"
            << " Тактов на одну операцию: " << (double) processor_clocks_independent / operations << "\n\n";
}

int main(int argc, char **argv) {
    uint64_t size = std::stoi(argv[1]);
    std::vector<double> first(size), second(size);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,100.0);
    for (std::size_t i = 0; i < size; ++i) {
        first[i] = distribution(generator);
        second[i] = distribution(generator);
    }

    std::cout << "Количество операций: " << size << "\n\n";

    benchmark(first, second);
    sse2_benchmark(first, second);
    avx2_benchmark(first, second);

    return EXIT_SUCCESS;
}