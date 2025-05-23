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

#include "Tuner1D.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <utility>

#include <sys/time.h>

#define STREAM_NTIMES 20

// using idx_t = size_t;
using real_t = double;
using val_t = Kokkos::complex<real_t>;
constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);

#define HLINE "-------------------------------------------------------------\n"

template <idx_t Nd, idx_t Nc>
using GaugeField = Kokkos::View<val_t* [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <idx_t Nc>
using SUNField = Kokkos::View<val_t* [Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

#if defined(KOKKOS_ENABLE_CUDA)
template <idx_t Nd, idx_t Nc>
using constGaugeField = Kokkos::View<const val_t* [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <idx_t Nc>
using constSUNField = Kokkos::View<const val_t* [Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <idx_t Nd, idx_t Nc>
using constGaugeField = Kokkos::View<const val_t* [Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <idx_t Nc>
using constSUNField = Kokkos::View<const val_t* [Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif

template <idx_t Nd, idx_t Nc>
using StreamHostArray = typename GaugeField<Nd, Nc>::HostMirror;

// forward declaration
template <idx_t Nc>
struct Matrix;

template <idx_t Nd, idx_t Nc>
struct MatrixRef;

template <idx_t Nd, idx_t Nc>
struct deviceGaugeField {
    deviceGaugeField() = delete;

    deviceGaugeField(idx_t n0, idx_t n1, idx_t n2, idx_t n3, const val_t init)
        : m_n({ n0, n1, n2, n3 })
    {
        do_init(n0 * n1 * n2 * n3, init);
    }

    // need to take care of 'this'-pointer capture
    void
    do_init(idx_t n, val_t init)
    {
        Kokkos::realloc(Kokkos::WithoutInitializing, view, n);

        // need a local copy of the view to have it captured below since this is a host function
        auto v = view;
        tune_and_launch_for(
            "init-GaugeField",
            0,
            n,
            KOKKOS_LAMBDA(const idx_t i) {
#pragma unroll
                for (idx_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
                    for (idx_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
                        for (idx_t c2 = 0; c2 < Nc; ++c2) {
                            v(i, mu, c1, c2) = init;
                        }
                    }
                }
            });
        Kokkos::fence();
    }

    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator()(idx_t i, idx_t mu, idx_t c1, idx_t c2) noexcept
    {
        return view(i, mu, c1, c2);
    }

    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator()(idx_t i, idx_t mu, idx_t c1, idx_t c2) const noexcept
    {
        return view(i, mu, c1, c2);
    }
    const Kokkos::Array<idx_t, 4> m_n;

    GaugeField<Nd, Nc> view;
};

struct deviceDoubleField {
    deviceDoubleField() = delete;

    deviceDoubleField(idx_t n0, idx_t n1, idx_t n2, idx_t n3, const double init)
        : m_n({ n0, n1, n2, n3 })
    {
        do_init(n0 * n1 * n2 * n3, init);
    }

    void
    do_init(idx_t n, const double init)
    {
        Kokkos::realloc(Kokkos::WithoutInitializing, view, n);
        // need a version of the view which can be captured since this is a host function
        auto v = view;
        tune_and_launch_for("init-DoubleField", 0, n, KOKKOS_LAMBDA(const idx_t i) { v(i) = init; });
        Kokkos::fence();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t i) noexcept
    {
        return view(i);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t i) const noexcept
    {
        return view(i);
    }

    const Kokkos::Array<idx_t, 4> m_n;

    Kokkos::View<double*> view;
};

template <idx_t Nd, idx_t Nc>
struct MatrixRef {
    MatrixRef() = delete;
    MatrixRef(const MatrixRef& other) = default;

    KOKKOS_FORCEINLINE_FUNCTION
    MatrixRef(idx_t i, idx_t mu, const deviceGaugeField<Nd, Nc>& g) noexcept
        : m_i(i)
        , m_mu(mu)
        , m_g(g)
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator()(idx_t c1, idx_t c2) const noexcept
    {
        return m_g(m_i, m_mu, c1, c2);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    auto&
    operator=(const MatrixRef& other) const noexcept
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
    set_i(idx_t i) { m_i = i; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_mu(idx_t mu) { m_mu = mu; }

    KOKKOS_FORCEINLINE_FUNCTION
    void
    set_idcs(idx_t i, idx_t mu)
    {
        m_i = i;
        m_mu = mu;
    }

    idx_t m_mu;
    idx_t m_i;
    const deviceGaugeField<Nd, Nc>& m_g;
};

template <idx_t Nc>
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

    template <idx_t Nd>
    KOKKOS_FORCEINLINE_FUNCTION auto&
    operator=(const MatrixRef<Nd, Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    conj(const MatrixRef<Nd, Nc>& m) noexcept
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

template <idx_t Nc>
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const MatrixRef<Nd, Nc>& lhs, const MatrixRef<Nd, Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator+(const MatrixRef<Nd, Nc>& lhs, const MatrixRef<Nd, Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator-(const MatrixRef<Nd, Nc>& lhs, const MatrixRef<Nd, Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const MatrixRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator+(const MatrixRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator-(const MatrixRef<Nd, Nc>& lhs, const Matrix<Nc>& rhs) noexcept
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

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
    Matrix<Nc>
    operator*(const Matrix<Nc>& lhs, const MatrixRef<Nd, Nc>& rhs) noexcept
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

template <idx_t Nc>
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

template <idx_t Nc>
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

template <idx_t Nc>
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

template <idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION double ReTr(const Matrix<Nc>& m) noexcept
{
    double rval = 0.0;
#pragma unroll
    for (idx_t c = 0; c < Nc; ++c) {
        rval += m(c, c).real();
    }
    return rval;
}

template <idx_t Nd, idx_t Nc>
KOKKOS_FORCEINLINE_FUNCTION double ReTr(const MatrixRef<Nd, Nc>& m) noexcept
{
    double rval = 0.0;
#pragma unroll
    for (idx_t c = 0; c < Nc; ++c) {
        rval += m(c, c).real();
    }
    return rval;
}

class geometry {
public:
    geometry() = delete;
    geometry(const geometry& other) = default;

    geometry(Kokkos::Array<idx_t, 4> n)
        : m_n(n)
        , m_nsites1(m_n[0])
        , m_nsites2(m_nsites1 * m_n[1])
        , m_nsites3(m_nsites2 * m_n[2])
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::Array<idx_t, 4>
    get_coords(const idx_t i) const noexcept
    {
        idx_t n0 = i % m_n[0];
        idx_t remaining = i / m_n[0];
        idx_t n1 = remaining % m_n[1];
        remaining /= m_n[1];
        idx_t n2 = remaining % m_n[2];
        idx_t n3 = remaining / m_n[2];

        return Kokkos::Array<idx_t, 4> { n0, n1, n2, n3 };
    }

    KOKKOS_FORCEINLINE_FUNCTION
    idx_t
    s0(const Kokkos::Array<idx_t, 4>& coords, const int delta) const noexcept
    {
        return ((coords[0] + delta + m_n[0]) % m_n[0]
            + coords[1] * m_nsites1
            + coords[2] * m_nsites2
            + coords[3] * m_nsites3);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    idx_t
    s1(const Kokkos::Array<idx_t, 4>& coords, const int delta) const noexcept
    {
        return (coords[0]
            + ((coords[1] + delta + m_n[1]) % m_n[1]) * m_nsites1
            + coords[2] * m_nsites2
            + coords[3] * m_nsites3);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    idx_t
    s2(const Kokkos::Array<idx_t, 4>& coords, const int delta) const noexcept
    {
        return (coords[0]
            + coords[1] * m_nsites1
            + ((coords[2] + delta + m_n[2]) % m_n[2]) * m_nsites2
            + coords[3] * m_nsites3);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    idx_t
    s3(const Kokkos::Array<idx_t, 4>& coords, const int delta) const noexcept
    {
        return (coords[0]
            + coords[1] * m_nsites1
            + coords[2] * m_nsites2
            + ((coords[3] + delta + m_n[3]) % m_n[3]) * m_nsites3);
    }

private:
    const Kokkos::Array<idx_t, 4> m_n;

    const idx_t m_nsites1;
    const idx_t m_nsites2;
    const idx_t m_nsites3;
};

template <idx_t Nd, idx_t Nc>
void perform_matmul(deviceGaugeField<Nd, Nc> a, deviceGaugeField<Nd, Nc> b,
    deviceGaugeField<Nd, Nc> c)
{
    using mref_t = MatrixRef<Nd, Nc>;
    using m_t = Matrix<Nc>;

    tune_and_launch_for(
        "su3xsu3",
        0,
        b.view.extent(0),
        KOKKOS_LAMBDA(const idx_t i) {
            mref_t A(i, 0, a);
            mref_t B(i, 0, b);
            mref_t C(i, 0, c);
            // unrolling this loop leads to cudaErrorIllegalAddress even if instead of using
            // set_mu one instantiates different instances of A, B and C for each mu
            for (idx_t mu = 0; mu < Nd; ++mu) {
                A.set_mu(mu);
                B.set_mu(mu);
                C.set_mu(mu);
                A = B * C;
            }
        });

    Kokkos::fence();
}

template <idx_t Nd, idx_t Nc>
void perform_conj_matmul(deviceGaugeField<Nd, Nc> a, deviceGaugeField<Nd, Nc> b,
    deviceGaugeField<Nd, Nc> c)
{
    using mref_t = MatrixRef<Nd, Nc>;
    using m_t = Matrix<Nc>;
    tune_and_launch_for(
        "conjsu3xsu3",
        0,
        b.view.extent(0),
        KOKKOS_LAMBDA(const idx_t i) {
            mref_t A(i, 0, a);
            mref_t B(i, 0, b);
            mref_t C(i, 0, c);
            // unrolling this loop leads to cudaErrorIllegalAddress even if instead of using
            // set_mu one instantiates different instances of A, B and C for each mu
            for (idx_t mu = 0; mu < Nd; ++mu) {
                A.set_mu(mu);
                B.set_mu(mu);
                C.set_mu(mu);
                A = conj(B) * C;
            }
        });

    Kokkos::fence();
}

template <idx_t Nd, idx_t Nc>
double perform_plaquette(deviceGaugeField<Nd, Nc> g, deviceDoubleField plaq_field)
{
    using mref_t = MatrixRef<Nd, Nc>;
    using m_t = Matrix<Nc>;

    double plaquette;

    const idx_t nsites = g.view.extent(0);

    geometry geom(g.m_n);

    tune_and_launch_for(
        "plaquette_kernel",
        0,
        nsites,
        KOKKOS_LAMBDA(const idx_t i) {
            const Kokkos::Array<idx_t, 4> coords = geom.get_coords(i);
            const idx_t ip0 = geom.s0(coords, 1);
            const idx_t ip1 = geom.s1(coords, 1);
            const idx_t ip2 = geom.s2(coords, 1);
            const idx_t ip3 = geom.s3(coords, 1);

            m_t tmp, lmu, lnu;

            // 0 1
            mref_t Umu(i, 0, g);
            mref_t Umunu(ip0, 1, g);
            mref_t Unu(i, 1, g);
            mref_t Unumu(ip1, 0, g);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) = ReTr(tmp);

            // 0 2
            Umunu.set_idcs(ip0, 2);
            Unu.set_mu(2);
            Unumu.set_idcs(ip2, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) += ReTr(tmp);

            // 1 2
            Umu.set_mu(1);
            Umunu.set_idcs(ip1, 2);
            Unumu.set_idcs(ip2, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) += ReTr(tmp);

            // 0 3
            Umu.set_mu(0);
            Unu.set_mu(3);
            Umunu.set_idcs(ip0, 3);
            Unumu.set_idcs(ip3, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) += ReTr(tmp);

            // 1 3
            Umu.set_mu(1);
            Umunu.set_idcs(ip2, 3);
            Unumu.set_idcs(ip3, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) += ReTr(tmp);

            // 2 3
            Umu.set_mu(2);
            Umunu.set_idcs(ip2, 3);
            Unumu.set_idcs(ip3, 2);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            plaq_field(i) += ReTr(tmp);
        });
    Kokkos::fence();
    // FIXME: this should also be auto-tuned but for that we need support for reductions in the tuner
    Kokkos::parallel_reduce(
        "plaquette reduction",
        nsites,
        KOKKOS_LAMBDA(const idx_t i, double& lplaq) { lplaq += plaq_field(i); },
        Kokkos::Sum<double>(plaquette));
    Kokkos::fence();
    return plaquette;
}

template <idx_t Nd, idx_t Nc>
double perform_plaquette_wreduce(deviceGaugeField<Nd, Nc> g)
{
    using mref_t = MatrixRef<Nd, Nc>;
    using m_t = Matrix<Nc>;

    double plaquette;

    const idx_t nsites = g.view.extent(0);

    geometry geom(g.m_n);

    // FIXME: this should also be auto-tuned but we need support for reductions in the tuner
    Kokkos::parallel_reduce(
        "plaquette_kernel",
        nsites,
        KOKKOS_LAMBDA(const idx_t i, double& lplaq) {
            const Kokkos::Array<idx_t, 4> coords = geom.get_coords(i);
            const idx_t ip0 = geom.s0(coords, 1);
            const idx_t ip1 = geom.s1(coords, 1);
            const idx_t ip2 = geom.s2(coords, 1);
            const idx_t ip3 = geom.s3(coords, 1);

            m_t tmp, lmu, lnu;

            // 0 1
            mref_t Umu(i, 0, g);
            mref_t Umunu(ip0, 1, g);
            mref_t Unu(i, 1, g);
            mref_t Unumu(ip1, 0, g);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);

            // 0 2
            Umunu.set_idcs(ip0, 2);
            Unu.set_mu(2);
            Unumu.set_idcs(ip2, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);

            // 1 2
            Umu.set_mu(1);
            Umunu.set_idcs(ip1, 2);
            Unumu.set_idcs(ip2, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);

            // 0 3
            Umu.set_mu(0);
            Unu.set_mu(3);
            Umunu.set_idcs(ip0, 3);
            Unumu.set_idcs(ip3, 0);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);

            // 1 3
            Umu.set_mu(1);
            Umunu.set_idcs(ip2, 3);
            Unumu.set_idcs(ip3, 1);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);

            // 2 3
            Umu.set_mu(2);
            Umunu.set_idcs(ip2, 3);
            Unumu.set_idcs(ip3, 2);
            lmu = Umu * Umunu;
            lnu = Unu * Unumu;
            tmp = lmu * conj(lnu);
            lplaq += ReTr(tmp);
        },
        Kokkos::Sum<double>(plaquette));
    Kokkos::fence();
    return plaquette;
}

template <idx_t Nd, idx_t Nc>
int run_benchmark(const idx_t Nx)
{
    printf("Reports fastest timing per kernel\n");
    printf("Creating Views...\n");

    const idx_t nelem = Nx * Nx * Nx * Nx;

    const double suN_nelem = double(nelem) * Nc * Nc;

    const double gauge_nelem = Nd * suN_nelem;

    printf("Memory Sizes:\n");
    printf("- Gauge Array Size:  %zu*%zu*%zu^4\n",
        Nd, Nc, Nx);
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
    double plaquetteWreduceTime = std::numeric_limits<double>::max();

    printf("Initializing Views...\n");

    deviceGaugeField<Nd, Nc> dev_a(Nx, Nx, Nx, Nx, ainit);
    deviceGaugeField<Nd, Nc> dev_b(Nx, Nx, Nx, Nx, binit);
    deviceGaugeField<Nd, Nc> dev_c(Nx, Nx, Nx, Nx, cinit);
    deviceDoubleField plaq_field(Nx, Nx, Nx, Nx, 0.0);

    printf("Starting benchmarking...\n");

    Kokkos::Timer timer;

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        perform_matmul(dev_a, dev_b, dev_c);
        matmulTime = std::min(matmulTime, timer.seconds());
    }

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        perform_conj_matmul(dev_a, dev_b, dev_c);
        conjMatmulTime = std::min(conjMatmulTime, timer.seconds());
    }

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        double plaq = perform_plaquette(dev_b, plaq_field);
        plaquetteTime = std::min(plaquetteTime, timer.seconds());
    }

    for (idx_t k = 0; k < STREAM_NTIMES; ++k) {
        timer.reset();
        double plaq = perform_plaquette_wreduce(dev_a);
        plaquetteWreduceTime = std::min(plaquetteWreduceTime, timer.seconds());
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

    printf("Plaquette-SingleLoop    %11.4f GB/s %11.4e s\n",
        1.0e-09 * 1.0 * (double)sizeof(val_t) * gauge_nelem / plaquetteWreduceTime,
        plaquetteWreduceTime);

    printf("\n"
           "Plaquette               %11.4f Gflop/s\n",
        1.0e-09 * nelem * 6.0 * // six plaquettes in 4D
            ((3 * 9 * (18 + 4)) + // three su3 multiplications
                (3)) // trace accumulation (our plaquette is complex but we're interested only in the real part)
            / plaquetteTime);

    printf("Plaquette-SingleLoop    %11.4f Gflop/s\n",
        1.0e-09 * nelem * 6.0 * // six plaquettes in 4D
            ((3 * 9 * (18 + 4)) + // three su3 multiplications
                (3)) // trace accumulation (our plaquette is complex but we're interested only in the real part)
            / plaquetteWreduceTime);

    printf(HLINE);

    return rc;
}

int parse_args(int argc, char** argv, idx_t& lattice_extent)
{
    // Defaults
    lattice_extent = 32;

    const std::string help_string = "  -n <N>, --nelements <N>\n"
                                    "     Create views containing [4][Nc][Nc]<N>^4 elements.\n"
                                    "     Default: N=32\n"
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
            lattice_extent = atoi(optarg);
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
    printf("Kokkos 4D GaugeField (mu static with a local reference type) 1D RangePolicy STREAM Benchmark\n");
    printf(HLINE);

    Kokkos::initialize(argc, argv);
    int rc;
    idx_t lattice_extent;
    rc = parse_args(argc, argv, lattice_extent);
    if (rc == 0) {
        rc = run_benchmark<4, 3>(lattice_extent);
    } else if (rc == -2) {
        // Don't return error code when called with "-h"
        rc = 0;
    }
    Kokkos::finalize();

    return rc;
}
