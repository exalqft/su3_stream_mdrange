// A very rudimentary tuning setup for tiling
// in Kokkos MDRangePolicy
// This is a work in progress, currently only works well
// for 4D fields and really does some kind of tuning only
// in cudaspace

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <fstream>
#include <unordered_map>

#include "utils.hpp"

// define how many times the kernel is run to tune
#ifndef TUNE_NTIMES
#define TUNE_NTIMES 8
#endif

using idx_t = int;

static constexpr const int verbosity = 1;
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
template <size_t rank>
struct TuningHashTable {
    std::unordered_map<std::string, Kokkos::Array<idx_t, rank>> table;
    void
    insert(const std::string key, const Kokkos::Array<idx_t, rank>& value)
    {
        table[key] = value;
    }
    Kokkos::Array<idx_t, rank>
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

// create global 4D, 3D and 2D hash tables
inline TuningHashTable<4> tuning_hash_table_4D;
inline TuningHashTable<3> tuning_hash_table_3D;
inline TuningHashTable<2> tuning_hash_table_2D;

enum class kernel_type {
    stream,
    stencil
};

template <size_t rank, class FunctorType>
void tune_and_launch_for(
    std::string functor_id,
    const Kokkos::Array<idx_t, rank>& start,
    const Kokkos::Array<idx_t, rank>& end,
    FunctorType&& functor,
    kernel_type ktype = kernel_type::stream)
{
    // launch kernel if tuning is disabled
    if (!perform_tuning) {
        const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end);
        Kokkos::parallel_for(policy, std::forward<FunctorType>(functor));
        return;
    }
    // create a unique string for the kernel
    std::string start_uid = "";
    std::string end_uid = "";
    for (size_t i = 0; i < rank; i++) {
        start_uid += std::to_string(start[i]) + (i == rank - 1 ? "" : "_");
        end_uid += std::to_string(end[i]) + (i == rank - 1 ? "" : "_");
    }
    const std::string functor_uid
        = functor_id + "_rank_" + std::to_string(rank) + "_start_" + start_uid + "_end_" + end_uid;

    if constexpr (rank == 4) {
        if (tuning_hash_table_4D.contains(functor_uid)) {
            const auto tiling = tuning_hash_table_4D.get(functor_uid);
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d %d %d\n",
                    functor_uid.c_str(),
                    tiling[0],
                    tiling[1],
                    tiling[2],
                    tiling[3]);
            }
            const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, tiling);
            Kokkos::parallel_for(policy, std::forward<FunctorType>(functor));
            return;
        }
    } else if constexpr (rank == 3) {
        if (tuning_hash_table_3D.contains(functor_uid)) {
            auto tiling = tuning_hash_table_3D.get(functor_uid);
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d %d %d\n",
                    functor_uid.c_str(),
                    tiling[0],
                    tiling[1],
                    tiling[2],
                    tiling[3]);
            }
            auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, tiling);
            Kokkos::parallel_for(policy, functor);
            return;
        }
    } else if constexpr (rank == 2) {
        if (tuning_hash_table_2D.contains(functor_uid)) {
            auto tiling = tuning_hash_table_2D.get(functor_uid);
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d %d %d\n",
                    functor_uid.c_str(),
                    tiling[0],
                    tiling[1],
                    tiling[2],
                    tiling[3]);
            }
            auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, tiling);
            Kokkos::parallel_for(policy, functor);
            return;
        }
    } else {
        // unsupported rank
        printf("Error: unsupported rank %d\n", rank);
        return;
    }
    if (verbosity > 2) {
        printf("Start tuning for kernel %s\n", functor_uid.c_str());
    }
    // if not tuned, tune the functor
    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end);
    Kokkos::Array<idx_t, rank> best_tiling;
    for (size_t i = 0; i < rank; i++) {
        best_tiling[i] = 1;
    }
    // timer for tuning
    Kokkos::Timer timer;
    double best_time = std::numeric_limits<double>::max();
    // first for hostspace
    // there is no tuning
    if constexpr (std::is_same_v<
                      typename Kokkos::DefaultExecutionSpace,
                      Kokkos::DefaultHostExecutionSpace>) {
        idx_t n_threads = Kokkos::DefaultExecutionSpace::concurrency();

        if (ktype != kernel_type::stencil) {
            // for OpenMP in stream-like kernels we try to find a tiling which prioritises
            // threading over the outer (leftmost) dimensions attempting to make the inner
            // dimensions as large as possible
            // In the process we try to find a tiling preventing threads from idling.
            for (size_t i = 0; i < rank; ++i) {
                idx_t curr_gcd = gcd(end[i] - start[i], n_threads);
                best_tiling[i] = (end[i] - start[i]) / curr_gcd;
                n_threads /= curr_gcd;
            }
        } else if (ktype == kernel_type::stencil) {
            // for stencil-like kernels we try to distribute the tiling more evenly in the hope
            // of promiting cache locality and avoiding unnecessary capacity misses
            // we don't take into account the size of the caches however
            const unsigned nrounds = 3;
            for (unsigned r = 0; r < nrounds; ++r) {
                for (size_t i = 0; i < rank; ++i) {
                    idx_t curr_lcd = lcd(end[i] - start[i], n_threads);
                    best_tiling[i] = r == 0 ? (end[i] - start[i]) / curr_lcd : best_tiling[i] / curr_lcd;
                    n_threads /= curr_lcd;
                }
            }
        }

        if (n_threads > 1) {
            if (verbosity > 0) {
                printf(
                    "[tune_and_launch_for] WARNING: Failed to find tiling "
                    "for kernel %s -> compromise: (XYZT) ",
                    functor_uid.c_str());
            }
            for (size_t i = 0; i < rank - 2; ++i) {
                best_tiling[i] = ((end[i] - start[i]) % 2 == 0) ? 2 : 1;
            }
            if (verbosity > 0) {
                for (size_t i = 0; i < rank - 1; ++i) {
                    printf("%d ", best_tiling[i]);
                }
                printf("%d\n", best_tiling[rank - 1]);
            }
        } else {
            if (verbosity > 1) {
                printf("[tune_and_launch_for] Tiling for kernel %s: (XYZT) ", functor_uid.c_str());
                for (size_t i = 0; i < rank - 1; ++i) {
                    printf("%d ", best_tiling[i]);
                }
                printf("%d\n", best_tiling[rank - 1]);
            }
        }
    } else {
        // for Cuda we need to tune the tiling
        const auto max_tile = policy.max_total_tile_size() / 2;
        Kokkos::Array<idx_t, rank> current_tiling;
        Kokkos::Array<idx_t, rank> tile_one;
        for (size_t i = 0; i < rank; i++) {
            current_tiling[i] = 1;
            best_tiling[i] = 1;
            tile_one[i] = 1;
        }
        std::vector<idx_t> fast_ind_tiles;
        idx_t fast_ind = max_tile;
        while (fast_ind > 2) {
            fast_ind = fast_ind / 2;
            fast_ind_tiles.push_back(fast_ind);
        }
        for (auto& tile : fast_ind_tiles) {
            current_tiling = tile_one;
            current_tiling[0] = tile;
            idx_t second_tile = max_tile / tile;
            while (second_tile > 1) {
                current_tiling[1] = second_tile;
                if (max_tile / tile / second_tile >= 4) {
                    for (size_t i : { 2, 1 }) {
                        current_tiling[2] = i;
                        current_tiling[3] = i;
                        auto tune_policy
                            = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, current_tiling);
                        double min_time = std::numeric_limits<double>::max();
                        for (int ii = 0; ii < TUNE_NTIMES; ii++) {
                            timer.reset();
                            Kokkos::parallel_for(tune_policy, functor);
                            Kokkos::fence();
                            min_time = std::min(min_time, timer.seconds());
                        }
                        if (min_time < best_time) {
                            best_time = min_time;
                            best_tiling = current_tiling;
                        }
                        if (verbosity > 2) {
                            printf(
                                "Current Tile size: %d %d %d %d, time: %11.4e\n",
                                current_tiling[0],
                                current_tiling[1],
                                current_tiling[2],
                                current_tiling[3],
                                min_time);
                        }
                    }
                } else if (max_tile / tile / second_tile == 2) {
                    for (int64_t i : { 2, 1 }) {
                        current_tiling[2] = i;
                        current_tiling[3] = 1;
                        auto tune_policy
                            = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, current_tiling);
                        double min_time = std::numeric_limits<double>::max();
                        for (int ii = 0; ii < TUNE_NTIMES; ii++) {
                            timer.reset();
                            Kokkos::parallel_for(tune_policy, functor);
                            Kokkos::fence();
                            min_time = std::min(min_time, timer.seconds());
                        }
                        if (min_time < best_time) {
                            best_time = min_time;
                            best_tiling = current_tiling;
                        }
                        if (verbosity > 2) {
                            printf(
                                "Current Tile size: %d %d %d %d, time: %11.4e\n",
                                current_tiling[0],
                                current_tiling[1],
                                current_tiling[2],
                                current_tiling[3],
                                min_time);
                        }
                    }
                } else {
                    current_tiling[2] = 1;
                    current_tiling[3] = 1;
                    auto tune_policy
                        = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, current_tiling);
                    double min_time = std::numeric_limits<double>::max();
                    for (int ii = 0; ii < TUNE_NTIMES; ii++) {
                        timer.reset();
                        Kokkos::parallel_for(tune_policy, functor);
                        Kokkos::fence();
                        min_time = std::min(min_time, timer.seconds());
                    }
                    if (min_time < best_time) {
                        best_time = min_time;
                        best_tiling = current_tiling;
                    }
                    if (verbosity > 2) {
                        printf(
                            "Current Tile size: %d %d %d %d, time: %11.4e\n",
                            current_tiling[0],
                            current_tiling[1],
                            current_tiling[2],
                            current_tiling[3],
                            min_time);
                    }
                }
                second_tile = second_tile / 2;
            }
        }
    }
    if (verbosity > 1) {
        printf(
            "Best Tile size for kernel %s: %d %d %d %d, time %.3e s\n",
            functor_uid.c_str(),
            best_tiling[0],
            best_tiling[1],
            best_tiling[2],
            best_tiling[3],
            best_time);
    }
    // store the best tiling in the hash table
    if constexpr (rank == 4) {
        tuning_hash_table_4D.insert(functor_uid, best_tiling);
    } else if constexpr (rank == 3) {
        tuning_hash_table_3D.insert(functor_uid, best_tiling);
    } else if constexpr (rank == 2) {
        tuning_hash_table_2D.insert(functor_uid, best_tiling);
    }
    if (verbosity > 3) {
        double time_rec = std::numeric_limits<double>::max();
        auto tune_policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end);
        for (int ii = 0; ii < TUNE_NTIMES; ii++) {
            timer.reset();
            Kokkos::parallel_for(tune_policy, functor);
            Kokkos::fence();
            time_rec = std::min(time_rec, timer.seconds());
        }
        printf("Time with default tile size: %11.4e s\n", time_rec);
        printf("Speedup: %f\n", time_rec / best_time);
    }
    const auto tune_policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(start, end, best_tiling);
    Kokkos::parallel_for(tune_policy, functor);
    return;
};

// write tune hash table to file
inline void
writeTuneCache(std::string cache_file_name)
{
    // open file in write mode
    std::ofstream cache_file(cache_file_name, std::ios::out);
    if (!cache_file.is_open()) {
        printf("Error: could not open cache file %s\n", cache_file_name.c_str());
        return;
    }
    // write the hash tables to the file
    for (const auto& entry : tuning_hash_table_4D.table) {
        cache_file << 4 << " " << entry.first << " ";
        for (const auto& value : entry.second) {
            cache_file << value << " ";
        }
        cache_file << "\n";
    }
    for (const auto& entry : tuning_hash_table_3D.table) {
        cache_file << 3 << " " << entry.first << " ";
        for (const auto& value : entry.second) {
            cache_file << value << " ";
        }
        cache_file << "\n";
    }
    for (const auto& entry : tuning_hash_table_2D.table) {
        cache_file << 2 << " " << entry.first << " ";
        for (const auto& value : entry.second) {
            cache_file << value << " ";
        }
        cache_file << "\n";
    }
    // close the file
    cache_file.close();
    if (verbosity > 0) {
        printf("Tuning hash table written to %s\n", cache_file_name.c_str());
    }
}

// read tune hash table from file
inline void
readTuneCache(std::string cache_file_name)
{
    // open file in read mode
    std::ifstream cache_file(cache_file_name);
    if (!cache_file.is_open()) {
        printf("Could not open cache file %s\n", cache_file_name.c_str());
        return;
    }
    // read the hash tables from the file
    std::string line;
    while (std::getline(cache_file, line)) {
        std::istringstream iss(line);
        size_t rank;
        std::string functor_id;
        iss >> rank;
        if (rank == 4) {
            iss >> functor_id;
            Kokkos::Array<idx_t, 4> tiling;
            for (size_t i = 0; i < 4; i++) {
                iss >> tiling[i];
            }
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d %d %d\n",
                    functor_id.c_str(),
                    tiling[0],
                    tiling[1],
                    tiling[2],
                    tiling[3]);
            }
            tuning_hash_table_4D.insert(functor_id, tiling);
        } else if (rank == 3) {
            iss >> functor_id;
            Kokkos::Array<idx_t, 3> tiling;
            for (size_t i = 0; i < 3; i++) {
                iss >> tiling[i];
            }
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d %d\n",
                    functor_id.c_str(),
                    tiling[0],
                    tiling[1],
                    tiling[2]);
            }
            tuning_hash_table_3D.insert(functor_id, tiling);
        } else if (rank == 2) {
            iss >> functor_id;
            Kokkos::Array<idx_t, 2> tiling;
            for (size_t i = 0; i < 2; i++) {
                iss >> tiling[i];
            }
            if (verbosity > 2) {
                printf(
                    "Tuning found for kernel %s, tiling: %d %d \n", functor_id.c_str(), tiling[0], tiling[1]);
            }
            tuning_hash_table_2D.insert(functor_id, tiling);
        } else {
            printf("Error: unsupported rank %zu\n", rank);
        }
    }
    // close the file
    cache_file.close();
    if (verbosity > 0) {
        printf("Tuning hash table read from %s\n", cache_file_name.c_str());
    }
}
