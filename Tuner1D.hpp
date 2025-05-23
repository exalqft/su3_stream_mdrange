// A very rudimentary tuning setup for tiling
// in Kokkos MDRangePolicy
// This is a work in progress, currently only works well
// for 4D fields and really does some kind of tuning only
// in cudaspace

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <unordered_map>

#include "utils.hpp"

// define how many times the kernel is run to tune
#ifndef TUNE_NTIMES
#define TUNE_NTIMES 8
#endif

using idx_t = int32_t;
using chunk_t = int32_t;

static constexpr const int verbosity = 2;
static constexpr const bool perform_tuning = true;

// need to hash the functor to get a unique id
// to store tuned tiling so that tuning is not repeated
// over multiple kernel calls
template <class FunctorType>
size_t
get_Functor_hash(const FunctorType& functor)
{
    // this is a very naive hash function
    // should be replaced with a better one
    return std::hash<const void*> {}(&functor);
}

// define a hash table to look up the tuned tiling
// this is a very naive hash table
// should be replaced with a better one
struct TuningHashTable {
    std::unordered_map<std::string, chunk_t> table;
    void
    insert(const std::string key, const chunk_t value)
    {
        table[key] = value;
    }

    chunk_t
    get(const std::string key)
    {
        return table[key];
    }

    bool
    contains(const std::string key)
    {
        return table.find(key) != table.end();
    }

    void
    clear()
    {
        table.clear();
    }
};

inline TuningHashTable tuning_hash_table;

enum class kernel_type {
    stream,
    stencil
};

template <class FunctorType>
void tune_and_launch_for(
    std::string functor_id,
    const idx_t start,
    const idx_t end,
    const FunctorType& functor,
    kernel_type ktype = kernel_type::stream)
{
    // launch kernel if tuning is disabled
    if (!perform_tuning) {
        Kokkos::RangePolicy<idx_t> policy(start, end);
        Kokkos::parallel_for(policy, functor);
        return;
    }
    // create a unique string for the kernel
    const std::string functor_uid
        = functor_id + "_start_" + std::to_string(start) + "_end_" + std::to_string(end);

    if (tuning_hash_table.contains(functor_uid)) {
        chunk_t chunk_size = tuning_hash_table.get(functor_uid);
        if (verbosity > 2) {
            printf(
                "Tuning found for kernel %s, chunk_size: %d\n",
                functor_uid.c_str(),
                chunk_size);
        }
        Kokkos::RangePolicy<idx_t> policy(start, end, Kokkos::ChunkSize { chunk_size });
        Kokkos::parallel_for(policy, functor);
        return;
    }
    if (verbosity > 2) {
        printf("Start tuning for kernel %s\n", functor_uid.c_str());
    }
    // if not tuned, tune the functor
    Kokkos::RangePolicy<idx_t> policy(start, end);
    chunk_t best_chunk_size = 1;
    // timer for tuning
    Kokkos::Timer timer;
    double best_time = std::numeric_limits<double>::max();
    // first for hostspace
    // there is no tuning
    if constexpr (std::is_same_v<
                      typename Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>) {
        idx_t n_threads = Kokkos::DefaultExecutionSpace::concurrency();

        // explicit narrowing conversion here if end-start is particularly large
        best_chunk_size = chunk_t((end - start) / n_threads);
        if (verbosity > 1) {
            printf("[tune_and_launch_for] Chunk size for kernel %s: %d\n", functor_uid.c_str(), best_chunk_size);
        }
    } else {
        // for GPUs we need to tune the chunk size
        // const auto max_chunk_size = policy.max_total_tile_size() / 2;
        const std::vector<chunk_t> chunk_sizes = {
            1, 2, 4, 6, 8, 12, 16, 18, 20, 21, 24, 28, 32,
            36, 40, 48, 50, 54, 56, 64, 72, 80, 96, 100, 112, 120, 128, 144,
            160, 180, 192, 200, 210, 224, 256, 288, 300, 320, 336, 360, 384, 400,
            432, 448, 512, 576, 600, 640, 720, 768, 864, 960, 1024
        };
        // vec.erase(
        //     std::remove_if(
        //         vec.begin(),
        //         vec.end(),
        //         [max_chunk_size](idx_t value) { return value > max_chunk_size; }),
        //     vec.end());

        for (chunk_t cs : chunk_sizes) {
            if ((end - start) % cs != 0) {
                continue;
            }
            Kokkos::RangePolicy<idx_t> tune_policy(start, end, Kokkos::ChunkSize { cs });
            double min_time = std::numeric_limits<double>::max();
            for (int i = 0; i < TUNE_NTIMES; ++i) {
                timer.reset();
                Kokkos::parallel_for(tune_policy, functor);
                Kokkos::fence();
                min_time = std::min(min_time, timer.seconds());
            }
            if (min_time < best_time) {
                best_time = min_time;
                best_chunk_size = cs;
            }
            if (verbosity > 2) {
                printf("Current chunk size: %d, time %3.6e\n", cs, min_time);
            }
        }
    }
    if (verbosity > 1) {
        printf(
            "Best chunk size for kernel %s: %d, time %.3e s\n",
            functor_uid.c_str(),
            best_chunk_size,
            best_time);
    }
    tuning_hash_table.insert(functor_uid, best_chunk_size);
    if (verbosity > 3) {
        double time_rec = std::numeric_limits<double>::max();
        Kokkos::RangePolicy<idx_t> tune_policy(start, end);
        for (int ii = 0; ii < TUNE_NTIMES; ii++) {
            timer.reset();
            Kokkos::parallel_for(tune_policy, functor);
            Kokkos::fence();
            time_rec = std::min(time_rec, timer.seconds());
        }
        printf("Time with default tile size: %11.4e s\n", time_rec);
        printf("Speedup: %f\n", time_rec / best_time);
    }
    Kokkos::RangePolicy<idx_t> tune_policy(start, end, Kokkos::ChunkSize { best_chunk_size });
    Kokkos::parallel_for(tune_policy, functor);
    return;
};
