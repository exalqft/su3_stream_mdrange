/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//
// Rephrasing as a benchmark for spinor linear algebra by Bartosz Kostrzewa (Uni Bonn)
//
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <utility>

#include <sys/time.h>

#include "Tuner.hpp"

#define STREAM_NTIMES 20

using real_t = double;
using val_t = Kokkos::complex<real_t>;
constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);
constexpr val_t dinit(1.4, 0.4);

// using val_t = double;
// constexpr val_t ainit(1.0);
// constexpr val_t binit(1.1);
// constexpr val_t cinit(1.3);

#define HLINE "-------------------------------------------------------------\n"

template <int Nd, int Nc>
using GaugeField = Kokkos::View<val_t**** [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using SUNField = Kokkos::View<val_t**** [Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

#if defined(KOKKOS_ENABLE_CUDA)
template <int Nd, int Nc>
using constGaugeField = Kokkos::View<const val_t**** [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <int Nc>
using constSUNField = Kokkos::View<const val_t**** [Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <int Nd, int Nc>
using constGaugeField = Kokkos::View<const val_t**** [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using constSUNField = Kokkos::View<const val_t**** [Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif

template <int Nd, int Nc>
using StreamHostArray = typename GaugeField<Nd, Nc>::HostMirror;

// forward declaration
template <int Nc>
struct Matrix;

template <int Nd, int Nc>
struct GaugeRef;

template <int rank>
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <std::size_t... Idcs>
constexpr Kokkos::Array<std::size_t, sizeof...(Idcs)>
make_repeated_sequence_impl(std::size_t value, std::integer_sequence<std::size_t, Idcs...>)
{
    return { ((void)Idcs, value)... };
}

template <std::size_t N>
constexpr Kokkos::Array<std::size_t, N>
make_repeated_sequence(std::size_t value)
{
    return make_repeated_sequence_impl(value, std::make_index_sequence<N> {});
}

template <typename V>
auto get_tiling(const V view)
{
    constexpr auto rank = view.rank_dynamic();
    // extract the dimensions from the view layout (assuming no striding)
    const auto& dimensions = view.layout().dimension;
    Kokkos::Array<size_t, rank> dims;
    for (int i = 0; i < rank; ++i) {
        dims[i] = dimensions[i];
    }
    // extract the recommended tiling for this view from a "default" policy
    const auto rec_tiling = Policy<rank>(make_repeated_sequence<rank>(0), dims).tile_size_recommended();

    if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>) {
        // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
        // for the innermost dimensions corresponds to the view extents
        return Kokkos::Array<idx_t, rank>({ 2, 2, view.extent_int(2), view.extent_int(3) });
    } else {
        // for GPUs we use the recommended tiling for now, we just need to convert it appropriately
        // from "array_index_type"
        // unfortunately the recommended tile size may exceed the maximum block size on GPUs
        // for large ranks -> let's cap the tiling at 4 dims
        constexpr auto max_rank = rank > 4 ? 4 : rank;
        Kokkos::Array<idx_t, max_rank> res;
        for (int i = 0; i < max_rank; ++i) {
            res[i] = rec_tiling[i];
        }
        return res;
    }
}

template <int Nd, int Nc>
struct deviceGaugeField {
    deviceGaugeField() = delete;

    deviceGaugeField(idx_t n0, idx_t n1, idx_t n2, idx_t n3, const val_t init)
    {
        do_init(n0, n1, n2, n3, init);
    }

    // need to take care of 'this'-pointer capture
    void
    do_init(idx_t n0, idx_t n1, idx_t n2, idx_t n3, val_t init)
    {
        using mref_t = GaugeRef<Nd, Nc>;
        Kokkos::realloc(Kokkos::WithoutInitializing, view, n0, n1, n2, n3);

        // need a local copy of the view to have it captured below since this is a host function
        auto v = view;
        // need a const view to get the constexpr rank
        const auto vconst = v;
        constexpr auto rank = vconst.rank_dynamic();

        tune_and_launch_for<4>(
            "init-GaugeField",
            { 0, 0, 0, 0 },
            { n0, n1, n2, n3 },
            KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
#pragma unroll
                for (int mu = 0; mu < Nd; ++mu) {
#pragma unroll
                    for (int c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
                        for (int c2 = 0; c2 < Nc; ++c2) {
                            v(i, j, k, l, mu, c1, c2) = init;
                        }
                    }
                }
            });
        Kokkos::fence();
    }

    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator()(idx_t i, idx_t j, idx_t k, idx_t l, idx_t mu, idx_t c1, idx_t c2) noexcept
    {
        return view(i, j, k, l, mu, c1, c2);
    }

    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator()(idx_t i, idx_t j, idx_t k, idx_t l, idx_t mu, idx_t c1, idx_t c2) const noexcept
    {
        return view(i, j, k, l, mu, c1, c2);
    }

    GaugeField<Nd, Nc> view;
};

struct deviceDoubleField {
    deviceDoubleField() = delete;

    deviceDoubleField(idx_t n0, idx_t n1, idx_t n2, idx_t n3, const double init)
    {
        do_init(n0, n1, n2, n3, init);
    }

    void
    do_init(idx_t n0, idx_t n1, idx_t n2, idx_t n3, const double init)
    {
        Kokkos::realloc(Kokkos::WithoutInitializing, view, n0, n1, n2, n3);

        // need a version of the view which can be captured since this is a host function
        auto v = view;
        // need a const view to get the constexpr rank
        const auto vconst = view;
        constexpr auto rank = vconst.rank_dynamic();
        const auto tiling = get_tiling(vconst);

        tune_and_launch_for<4>("init-DoubleField", { 0, 0, 0, 0 }, { n0, n1, n2, n3 }, KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l) { v(i, j, k, l) = init; });
        Kokkos::fence();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t i, idx_t j, idx_t k, idx_t l) noexcept
    {
        return view(i, j, k, l);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t i, idx_t j, idx_t k, idx_t l) const noexcept
    {
        return view(i, j, k, l);
    }

    Kokkos::View<double****> view;
};

template <int Nd, int Nc>
struct GaugeRef {
    GaugeRef() = delete;
    GaugeRef(const GaugeRef& other) = default;

    KOKKOS_FORCEINLINE_FUNCTION
    GaugeRef(idx_t x, idx_t y, idx_t z, idx_t t, idx_t mu, const deviceGaugeField<Nd, Nc>& g) noexcept
        : m_x(x)
        , m_y(y)
        , m_z(z)
        , m_t(t)
        , m_mu(mu)
        , m_g(g)
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t c1, idx_t c2) const noexcept
    {
        return m_g(m_x, m_y, m_z, m_t, m_mu, c1, c2);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator=(const GaugeRef& other) const noexcept
    {
#pragma unroll
        for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (idx_t c2 = 0; c2 < Nc; ++c2) {
                (*this)(c1, c2) = other(c1, c2);
            }
        }
        return *this;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator=(const Matrix<Nc>& other) const noexcept
    {
#pragma unroll
        for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (idx_t c2 = 0; c2 < Nc; ++c2) {
                (*this)(c1, c2) = other(c1, c2);
            }
        }
        return *this;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_x(idx_t x) { m_x = x; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_y(idx_t y) { m_y = y; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_z(idx_t z) { m_z = z; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_t(idx_t t) { m_t = t; }
    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_mu(idx_t mu) { m_mu = mu; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_idcs(idx_t x, idx_t y, idx_t z, idx_t t, idx_t mu)
    {
        m_x = x;
        m_y = y;
        m_z = z;
        m_t = t;
        m_mu = mu;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_idcs(idx_t x, idx_t y, idx_t z, idx_t t)
    {
        m_x = x;
        m_y = y;
        m_z = z;
        m_t = t;
    }

    idx_t m_mu;
    idx_t m_x;
    idx_t m_y;
    idx_t m_z;
    idx_t m_t;
    const deviceGaugeField<Nd, Nc>& m_g;
};

template <int Nc>
struct Matrix : public Kokkos::Array<Kokkos::Array<val_t, size_t(Nc)>, size_t(Nc)> {
    Matrix() = default;

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t c1, idx_t c2) noexcept
    {
        return (*this)[c1][c2];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    const auto&
    operator()(idx_t c1, idx_t c2) const noexcept
    {
        return (*this)[c1][c2];
    }

    template <int Nd>
    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator=(const GaugeRef<Nd, Nc>& rhs) noexcept
    {
#pragma unroll
        for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (idx_t c2 = 0; c2 < Nc; ++c2) {
                (*this)(c1, c2) = rhs(c1, c2);
            }
        }
        return *this;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator=(const Matrix& other) noexcept
    {
#pragma unroll
        for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (idx_t c2 = 0; c2 < Nc; ++c2) {
                (*this)(c1, c2) = other(c1, c2);
            }
        }
        return *this;
    }
};

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    conj(const GaugeRef<Nd, Nc>& m) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = Kokkos::conj(m(c2, c1));
        }
    }
    return out;
}

template <int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    conj(const Matrix<Nc>& m) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = Kokkos::conj(m(c2, c1));
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const GaugeRef<Nd, Nc>& lhs, const GaugeRef<Nd, Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, 0) * rhs(0, c2);
#pragma unroll
            for (idx_t c3 = 1; c3 < Nc; ++c3) {
                out(c1, c2) += lhs(c1, c3) * rhs(c3, c2);
            }
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator+(const GaugeRef<Nd, Nc>& lhs, const GaugeRef<Nd, Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) + rhs(c1, c2);
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator-(const GaugeRef<Nd, Nc>& lhs, const GaugeRef<Nd, Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) - rhs(c1, c2);
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const GaugeRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, 0) * rhs(0, c2);
#pragma unroll
            for (idx_t c3 = 1; c3 < Nc; ++c3) {
                out(c1, c2) += lhs(c1, c3) * rhs(c3, c2);
            }
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator+(const GaugeRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) + rhs(c1, c2);
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator-(const GaugeRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) - rhs(c1, c2);
        }
    }
    return out;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const Matrix<Nc>& lhs, const GaugeRef<Nd, Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, 0) * rhs(0, c2);
#pragma unroll
            for (idx_t c3 = 1; c3 < Nc; ++c3) {
                out(c1, c2) += lhs(c1, c3) * rhs(c3, c2);
            }
        }
    }
    return out;
}

template <int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const Matrix<Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, 0) * rhs(0, c2);
#pragma unroll
            for (idx_t c3 = 1; c3 < Nc; ++c3) {
                out(c1, c2) += lhs(c1, c3) * rhs(c3, c2);
            }
        }
    }
    return out;
}

template <int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator+(const Matrix<Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) + rhs(c1, c2);
        }
    }
    return out;
}

template <int Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator-(const Matrix<Nc>& lhs, const Matrix<Nc>& rhs) noexcept
{
    Matrix<Nc> out;
#pragma unroll
    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (idx_t c2 = 0; c2 < Nc; ++c2) {
            out(c1, c2) = lhs(c1, c2) - rhs(c1, c2);
        }
    }
    return out;
}

template <int Nc>
KOKKOS_FORCEINLINE_FUNCTION double ReTr(const Matrix<Nc>& m) noexcept
{
    double rval = 0.0;
#pragma unroll
    for (idx_t c = 0; c < Nc; ++c) {
        rval += m(c, c).real();
    }
    return rval;
}

template <int Nd, int Nc>
KOKKOS_FORCEINLINE_FUNCTION double ReTr(const GaugeRef<Nd, Nc>& m) noexcept
{
    double rval = 0.0;
#pragma unroll
    for (idx_t c = 0; c < Nc; ++c) {
        rval += m(c, c).real();
    }
    return rval;
}

template <int Nd, int Nc>
void perform_matmul(deviceGaugeField<Nd, Nc> a, deviceGaugeField<Nd, Nc> b,
    deviceGaugeField<Nd, Nc> c)
{
    using mref_t = GaugeRef<Nd, Nc>;
    using m_t = Matrix<Nc>;
    constexpr auto rank = a.view.rank_dynamic();
    const idx_t stream_array_size = a.view.extent_int(0);
    const auto tiling = get_tiling(a.view);

    tune_and_launch_for<rank>(
        "su3xsu3",
        { 0, 0, 0, 0 },
        { stream_array_size, stream_array_size, stream_array_size, stream_array_size },
        KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
            mref_t A(i, j, k, l, 0, a);
            mref_t B(i, j, k, l, 0, b);
            mref_t C(i, j, k, l, 0, c);
            // unrolling this loop leads to cudaErrorIllegalAddress even if instead of using
            // set_mu one instantiates different instances of A, B and C for each mu
            for (int mu = 0; mu < Nd; ++mu) {
                A.set_mu(mu);
                B.set_mu(mu);
                C.set_mu(mu);
                A = B * C;
            }
        });

    Kokkos::fence();
}

template <int Nd, int Nc>
void perform_conj_matmul_tmp(deviceGaugeField<Nd, Nc> a, deviceGaugeField<Nd, Nc> b,
    deviceGaugeField<Nd, Nc> c)
{
    using mref_t = GaugeRef<Nd, Nc>;
    using m_t = Matrix<Nc>;
    constexpr auto rank = a.view.rank_dynamic();
    const idx_t stream_array_size = a.view.extent_int(0);
    const auto tiling = get_tiling(a.view);
    tune_and_launch_for<rank>(
        "conjsu3xsu3",
        { 0, 0, 0, 0 },
        { stream_array_size, stream_array_size, stream_array_size, stream_array_size },
        KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
            mref_t A(i, j, k, l, 0, a);
            mref_t B(i, j, k, l, 0, b);
            mref_t C(i, j, k, l, 0, c);
            // unrolling this loop leads to cudaErrorIllegalAddress even if instead of using
            // set_mu one instantiates different instances of A, B and C for each mu
            for (int mu = 0; mu < Nd; ++mu) {
                A.set_mu(mu);
                B.set_mu(mu);
                C.set_mu(mu);
                A = conj(B) * C;
            }
        });

    Kokkos::fence();
}

template <int Nd, int Nc>
double perform_plaquette(deviceGaugeField<Nd, Nc> g, deviceDoubleField plaq_field)
{
    using mref_t = GaugeRef<Nd, Nc>;
    using m_t = Matrix<Nc>;
    constexpr auto rank = g.view.rank_dynamic();
    const idx_t stream_array_size = g.view.extent_int(0);
    const auto tiling = get_tiling(g.view);

    double plaquette;

    const idx_t N0 = g.view.extent_int(0);
    const idx_t N1 = g.view.extent_int(1);
    const idx_t N2 = g.view.extent_int(2);
    const idx_t N3 = g.view.extent_int(3);

    tune_and_launch_for<4>(
        "plaquette_kernel",
        { 0, 0, 0, 0 },
        { N0, N1, N2, N3 },
        KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
            idx_t ip = (i + 1) % N0;
            idx_t jp = (j + 1) % N1;
            idx_t kp = (k + 1) % N2;
            idx_t lp = (l + 1) % N3;

            m_t tmp, lmu, lnu;

            // 0 1
            mref_t Umu(i, j, k, l, 0, g);
            mref_t Umunu(ip, j, k, l, 1, g);
            mref_t Unu(i, j, k, l, 1, g);
            mref_t Unumu(i, jp, k, l, 0, g);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) = ReTr(tmp);

            // 0 2
            Umunu.set_idcs(ip, j, k, l, 2);
            Unu.set_mu(2);
            Unumu.set_idcs(i, j, kp, l, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) += ReTr(tmp);

            // 1 2
            Umu.set_mu(1);
            Umunu.set_idcs(i, jp, k, l, 2);
            Unumu.set_idcs(i, k, kp, l, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) += ReTr(tmp);

            // 0 3
            Umu.set_mu(0);
            Unu.set_mu(3);
            Umunu.set_idcs(ip, j, k, l, 3);
            Unumu.set_idcs(i, j, k, lp, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) += ReTr(tmp);

            // 1 3
            Umu.set_mu(1);
            Umunu.set_idcs(i, jp, k, l, 3);
            Unumu.set_idcs(i, j, k, lp, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) += ReTr(tmp);

            // 2 3
            Umu.set_mu(2);
            Umunu.set_idcs(i, j, kp, l, 3);
            Unumu.set_idcs(i, j, k, lp, 2);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i, j, k, l) += ReTr(tmp);
        });
    Kokkos::fence();
    // FIXME: this should also be auto-tuned but for that we need support for reductions in the tuner
    Kokkos::parallel_reduce(
        "plaquette reduction",
        Policy<4>({ 0, 0, 0, 0 }, { N0, N1, N2, N3 }, tiling),
        KOKKOS_LAMBDA(const idx_t i, const idx_t j, const idx_t k, const idx_t l, double& lplaq) { lplaq += plaq_field(i, j, k, l); },
        Kokkos::Sum<double>(plaquette));
    Kokkos::fence();
    return plaquette;
}

template <int Nd, int Nc>
int run_benchmark(const idx_t stream_array_size)
{
    printf("Reports fastest timing per kernel\n");
    printf("Creating Views...\n");

    const double nelem = (double)stream_array_size * (double)stream_array_size * (double)stream_array_size * (double)stream_array_size;

    const double suN_nelem = nelem * Nc * Nc;

    const double gauge_nelem = Nd * suN_nelem;

    printf("Memory Sizes:\n");
    printf("- Gauge Array Size:  %d*%d*%" PRIu64 "^4\n",
        Nd, Nc,
        static_cast<uint64_t>(stream_array_size));
    printf("- Per SUNField:          %12.2f MB\n",
        1.0e-6 * suN_nelem * (double)sizeof(val_t));
    printf("- Total:                 %12.2f MB\n",
        1.0e-6 * (suN_nelem + 3.0 * gauge_nelem) * (double)sizeof(val_t));

    printf("Benchmark kernels will be performed for %d iterations.\n",
        STREAM_NTIMES);

    printf(HLINE);

    double matmulTime = std::numeric_limits<double>::max();
    double conjMatmulTime = std::numeric_limits<double>::max();
    double plaquetteTime = std::numeric_limits<double>::max();

    printf("Initializing Views...\n");

    deviceGaugeField<Nd, Nc> dev_a(stream_array_size, stream_array_size, stream_array_size, stream_array_size, ainit);
    deviceGaugeField<Nd, Nc> dev_b(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit);
    deviceGaugeField<Nd, Nc> dev_c(stream_array_size, stream_array_size, stream_array_size, stream_array_size, cinit);
    deviceDoubleField plaq_field(stream_array_size, stream_array_size, stream_array_size, stream_array_size, 0.0);

    printf("Starting benchmarking...\n");

    Kokkos::Timer timer;

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        perform_matmul(dev_a, dev_b, dev_c);
        matmulTime = std::min(matmulTime, timer.seconds());
    }

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        perform_conj_matmul_tmp(dev_a, dev_b, dev_c);
        conjMatmulTime = std::min(conjMatmulTime, timer.seconds());
    }

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        double plaq = perform_plaquette(dev_b, plaq_field);
        plaquetteTime = std::min(plaquetteTime, timer.seconds());
    }

    int rc = 0;

    printf(HLINE);

    printf("MatMul                  %11.4f GB/s\n",
        1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / matmulTime);

    printf("conjMatMul              %11.4f GB/s\n",
        1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / conjMatmulTime);

    printf("Plaquette               %11.4f GB/s %11.4e s\n",
        1.0e-09 * 1.0 * (double)sizeof(val_t) * gauge_nelem / plaquetteTime,
        plaquetteTime);

    printf("\n"
           "Plaquette               %11.4f Gflop/s\n",
        1.0e-09 * nelem * 6.0 * // six plaquettes in 4D
            ((2 * 9 * (18 + 4)) + // two su3 multiplications
                (3 * (18 + 4)) + // one su3 multiplcation (diagonal elements only)
                (3)) // trace accumulation (our plaquette is complex but we're interested only in the real part)
            / plaquetteTime);

    printf(HLINE);

    return rc;
}

int parse_args(int argc, char** argv, idx_t& stream_array_size)
{
    // Defaults
    stream_array_size = 32;

    const std::string help_string = "  -n <N>, --nelements <N>\n"
                                    "     Create stream views containing [4][Nc][Nc]<N>^4 elements.\n"
                                    "     Default: 32\n"
                                    "  -h, --help\n"
                                    "     Prints this message.\n"
                                    "     Hint: use --kokkos-help to see command line options provided by "
                                    "Kokkos.\n";

    static struct option long_options[] = {
        { "nelements", required_argument, NULL, 'n' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) != -1)
        switch (c) {
        case 'n':
            stream_array_size = atoi(optarg);
            break;
        case 'h':
            printf("%s", help_string.c_str());
            return -2;
            break;
        case 0:
            break;
        default:
            printf("%s", help_string.c_str());
            return -1;
            break;
        }
    return 0;
}

int main(int argc, char* argv[])
{
    printf(HLINE);
    printf("Kokkos 4D GaugeField (mu static with a local reference type) MDRangePolicy STREAM Benchmark\n");
    printf(HLINE);

    Kokkos::initialize(argc, argv);
    int rc;
    idx_t stream_array_size;
    rc = parse_args(argc, argv, stream_array_size);
    if (rc == 0) {
        rc = run_benchmark<4, 3>(stream_array_size);
    } else if (rc == -2) {
        // Don't return error code when called with "-h"
        rc = 0;
    }
    Kokkos::finalize();

    return rc;
}
