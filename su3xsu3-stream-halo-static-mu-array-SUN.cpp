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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>


#define STREAM_NTIMES 20

using val_t = Kokkos::complex<double>;

constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);
constexpr val_t dinit(1.4, 0.4);

#define HLINE "-------------------------------------------------------------\n"

template <int Nc>
using SUN =
  Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc>;

template <int Nd, int Nc>
using GaugeField =
    Kokkos::View<SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using SUNField =
    Kokkos::View<SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field =
    Kokkos::View<val_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const SUN<Nc> & a, const SUN<Nc> & b) {
  SUN<Nc> out;
  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      out[c1][c2] = a[c1][0] * b[0][c2];
      #pragma unroll
      for(int ci = 1; ci < Nc; ++ci){
        out[c1][c2] += a[c1][ci] * b[ci][c2];
      }
    }
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> conj(const SUN<Nc> & a) {
  SUN<Nc> out;
  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      out[c1][c2] = Kokkos::conj(a[c2][c1]);
    }
  }
  return out;
}

#if defined(KOKKOS_ENABLE_CUDA)
template <int Nd, int Nc>
using constGaugeField =
    Kokkos::View<const SUN<Nc> ****[Nd], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <int Nc>
using constSUNField =
    Kokkos::View<const SUN<Nc> ****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField =
    Kokkos::View<const val_t ****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <int Nd, int Nc>
using constGaugeField =
    Kokkos::View<const SUN<Nc> ****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using constSUNField =
    Kokkos::View<const SUN<Nc> ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField =
    Kokkos::View<const val_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif
template <int Nd, int Nc>
using StreamHostArray = typename GaugeField<Nd,Nc>::HostMirror;

using StreamIndex = int;

template <int rank>
using Policy      = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <std::size_t... Idcs>
constexpr Kokkos::Array<std::size_t, sizeof...(Idcs)>
make_repeated_sequence_impl(std::size_t value, std::integer_sequence<std::size_t, Idcs...>)
{
  return { ((void)Idcs, value)... };
}

template <std::size_t N>
constexpr Kokkos::Array<std::size_t,N> 
make_repeated_sequence(std::size_t value)
{
  return make_repeated_sequence_impl(value, std::make_index_sequence<N>{});
}

template <typename V>
auto
get_tiling(const V view)
{
  constexpr auto rank = view.rank_dynamic();
  // extract the dimensions from the view layout (assuming no striding)
  const auto & dimensions = view.layout().dimension;
  Kokkos::Array<std::size_t,rank> dims;
  for(int i = 0; i < rank; ++i){
    dims[i] = dimensions[i];
  }
  // extract the recommended tiling for this view from a "default" policy 
  const auto rec_tiling = Policy<rank>(make_repeated_sequence<rank>(0),dims).tile_size_recommended();
  
  if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>){
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
    // for the innermost dimensions corresponds to the view extents
    return Kokkos::Array<std::size_t,rank>({1,1,view.extent(2),view.extent(3)});
  } else {
    // for GPUs we use the recommended tiling for now, we just need to convert it appropriately
    // from "array_index_type"
    // unfortunately the recommended tile size may exceed the maximum block size on GPUs 
    // for large ranks -> let's cap the tiling at 4 dims
    constexpr auto max_rank = rank > 4 ? 4 : rank;
    Kokkos::Array<std::size_t,max_rank> res;
    for(int i = 0; i < max_rank; ++i){
      res[i] = rec_tiling[i];
    }
    return res;
  }
}

template <int Nd, int Nc>
struct deviceGaugeField {
  deviceGaugeField() = delete;

  deviceGaugeField(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init)
  {
    do_init(N0,N1,N2,N3,view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          GaugeField<Nd,Nc> & V, const val_t init){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const GaugeField<Nd,Nc> vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              V(i,j,k,l,mu)[c1][c2] = init;
            }
          }
        }
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l, const int mu) const {
    return view(i,j,k,l,mu);
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l, const int mu) {
    return view(i,j,k,l,mu);
  }

  GaugeField<Nd,Nc> view;
};

template <int Nc>
struct deviceSUNField {
  deviceSUNField() = delete;

  deviceSUNField(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init)
  {
    do_init(N0,N1,N2,N3,view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          SUNField<Nc> & V, const val_t init){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const SUNField<Nc> vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        #pragma unroll
        for(int c1 = 0; c1 < Nc; ++c1){
          #pragma unroll
          for(int c2 = 0; c2 < Nc; ++c2){
            V(i,j,k,l)[c1][c2] = init;
          }
        }
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    return view(i,j,k,l);
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    return view(i,j,k,l);
  }

  SUNField<Nc> view;
};

struct deviceField {
  deviceField() = delete;

  deviceField(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init)
  {
    do_init(N0,N1,N2,N3,view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          Field & V, const val_t init){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const Field vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        V(i,j,k,l) = init;
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    return view(i,j,k,l);
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    return view(i,j,k,l);
  }

  Field view;
};


template <int Nd, int Nc>
struct deviceGaugeField_h {
  deviceGaugeField_h() = delete;

  deviceGaugeField_h(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init,
                   const Kokkos::Array<int,4> dims_partitioned = {0,0,0,0}) 
                   : dims({N0,N1,N2,N3}), shifts(dims_partitioned)  
  {
    do_init(view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(GaugeField<Nd,Nc> & V, const val_t init){
    Kokkos::Array<std::size_t,4> extents;
    for(int i = 0; i < 4; ++i) {
      extents[i] = dims[i] + 2*shifts[i];
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, extents[0], extents[1], extents[2], extents[3]);
    
    // need a const view to get the constexpr rank
    const GaugeField<Nd,Nc> vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), dims, tiling),
      KOKKOS_CLASS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        const StreamIndex ii = i + this->shifts[0];
        const StreamIndex jj = j + this->shifts[1];
        const StreamIndex kk = k + this->shifts[2];
        const StreamIndex ll = l + this->shifts[3];
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              V(ii,jj,kk,ll,mu)[c1][c2] = init;
            }
          }
        }
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l, const int mu) const {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll,mu);
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l, const int mu) {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll,mu);
  }

  GaugeField<Nd,Nc> view;
  const Kokkos::Array<std::size_t,4> dims;
  const Kokkos::Array<int,4> shifts;
};

template <int Nc>
struct deviceSUNField_h {
  deviceSUNField_h() = delete;

  deviceSUNField_h(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init,
                 const Kokkos::Array<int,4> dims_partitioned = {0,0,0,0}) 
                 : dims({N0,N1,N2,N3}), shifts(dims_partitioned)  
  {
    do_init(view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(SUNField<Nc> & V, const val_t init){
    Kokkos::Array<std::size_t,4> extents;
    for(int i = 0; i < 4; ++i) {
      extents[i] = dims[i] + 2*shifts[i];
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, extents[0], extents[1], extents[2], extents[3]);
    
    // need a const view to get the constexpr rank
    const SUNField<Nc> vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), dims, tiling),
      KOKKOS_CLASS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        const StreamIndex ii = i + this->shifts[0];
        const StreamIndex jj = j + this->shifts[1];
        const StreamIndex kk = k + this->shifts[2];
        const StreamIndex ll = l + this->shifts[3];
        #pragma unroll
        for(int c1 = 0; c1 < Nc; ++c1){
          #pragma unroll
          for(int c2 = 0; c2 < Nc; ++c2){
            V(ii,jj,kk,ll)[c1][c2] = init;
          }
        }
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll);
  }

  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll);
  }

  SUNField<Nc> view;
  const Kokkos::Array<std::size_t,4> dims;
  const Kokkos::Array<int,4> shifts;
};

struct deviceField_h {
  deviceField_h() = delete;

  deviceField_h(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init,
              const Kokkos::Array<int,4> dims_partitioned = {0,0,0,0}) 
              : dims({N0,N1,N2,N3}), shifts(dims_partitioned)  
  {
    do_init(view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(Field & V, const val_t init){
    Kokkos::Array<std::size_t,4> extents;
    for(int i = 0; i < 4; ++i) {
      extents[i] = dims[i] + 2*shifts[i];
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, extents[0], extents[1], extents[2], extents[3]);
    
    // need a const view to get the constexpr rank
    const Field vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), dims, tiling),
      KOKKOS_CLASS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        const StreamIndex ii = i + this->shifts[0];
        const StreamIndex jj = j + this->shifts[1];
        const StreamIndex kk = k + this->shifts[2];
        const StreamIndex ll = l + this->shifts[3];
        V(ii,jj,kk,ll) = init;
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll);
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    const StreamIndex ii = i + shifts[0];
    const StreamIndex jj = j + shifts[1];
    const StreamIndex kk = k + shifts[2];
    const StreamIndex ll = l + shifts[3];
    return view(ii,jj,kk,ll);
  }

  Field view;
  const Kokkos::Array<std::size_t,4> dims;
  const Kokkos::Array<int,4> shifts;
};

int parse_args(int argc, char **argv, StreamIndex &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create stream views containing [4][Nc][Nc]<N>^4 elements.\n"
      "     Default: 32\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

template <int Nd, int Nc>
void su3Xsu3(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
                    const deviceGaugeField<Nd,Nc> c) {
  constexpr auto rank = a.view.rank_dynamic();
  const auto stream_array_size = a.view.extent(0);
  const auto tiling = get_tiling(a.view);
  Kokkos::parallel_for(
      "su3xsu3", 
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          a(i,j,k,l,mu) = b(i,j,k,l,mu) * c(i,j,k,l,mu);
        }
      });
  Kokkos::fence();
}

template <int Nd, int Nc>
void su3Xsu3_h(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
                    const deviceGaugeField_h<Nd,Nc> c) {
  constexpr auto rank = a.view.rank_dynamic();
  const auto stream_array_size = a.view.extent(0);
  const auto tiling = get_tiling(a.view);
  Kokkos::parallel_for(
      "su3xsu3", 
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          a(i,j,k,l,mu) = b(i,j,k,l,mu) * c(i,j,k,l,mu);
        }
      });
  Kokkos::fence();
}

template <int Nd, int Nc>
int run_benchmark(const StreamIndex stream_array_size) {
  printf("Reports fastest timing per kernel\n");
  printf("Creating Views...\n");
  
  const double nelem = (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size;

  const double suN_nelem = nelem*Nc*Nc;

  const double gauge_nelem = Nd*suN_nelem;

  printf("Memory Sizes:\n");
  printf("- Gauge Array Size:  %d*%d*%" PRIu64 "^4\n",
         Nd, Nc,
         static_cast<uint64_t>(stream_array_size));
  printf("- Per SUNField:          %12.2f MB\n",
         1.0e-6 * suN_nelem * (double)sizeof(val_t));
  printf("- Total:                 %12.2f MB\n",
         1.0e-6 * (suN_nelem + 3.0*gauge_nelem) * (double)sizeof(val_t));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  double su3Xsu3_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_no_shift_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_x_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_y_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_z_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_t_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xy_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xz_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xt_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xyz_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xyt_Time  = std::numeric_limits<double>::max();
  double su3Xsu3_h_shift_xyzt_Time  = std::numeric_limits<double>::max();

  printf("Initializing Views...\n");

  deviceGaugeField<Nd,Nc> dev_a(stream_array_size,stream_array_size,stream_array_size,stream_array_size,ainit);
  deviceGaugeField<Nd,Nc> dev_b(stream_array_size,stream_array_size,stream_array_size,stream_array_size,binit);
  deviceGaugeField<Nd,Nc> dev_c(stream_array_size,stream_array_size,stream_array_size,stream_array_size,cinit);
  deviceGaugeField_h<Nd,Nc> dev_d_h_no_shift(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{0,0,0,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_x(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,0,0,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_y(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{0,1,0,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_z(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{0,0,1,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_t(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{0,0,0,1});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xy(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,1,0,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xz(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,0,1,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xt(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,0,0,1});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xyz(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,1,1,0});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xyt(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,0,1,1});
  deviceGaugeField_h<Nd,Nc> dev_d_h_shift_xyzt(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit,{1,1,1,1});

  printf("Starting Benchmark...\n");

  Kokkos::Timer timer;

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k){
    timer.reset();
    su3Xsu3(dev_a, dev_b, dev_c);
    su3Xsu3_Time = std::min(su3Xsu3_Time, timer.seconds()); 

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_no_shift);
    su3Xsu3_h_no_shift_Time = std::min(su3Xsu3_h_no_shift_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_x);
    su3Xsu3_h_shift_x_Time = std::min(su3Xsu3_h_shift_x_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_y);
    su3Xsu3_h_shift_y_Time = std::min(su3Xsu3_h_shift_y_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_z);
    su3Xsu3_h_shift_z_Time = std::min(su3Xsu3_h_shift_z_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_t);
    su3Xsu3_h_shift_t_Time = std::min(su3Xsu3_h_shift_t_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xy);
    su3Xsu3_h_shift_xy_Time = std::min(su3Xsu3_h_shift_xy_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xz);
    su3Xsu3_h_shift_xz_Time = std::min(su3Xsu3_h_shift_xz_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xt);
    su3Xsu3_h_shift_xt_Time = std::min(su3Xsu3_h_shift_xt_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xyz);
    su3Xsu3_h_shift_xyz_Time = std::min(su3Xsu3_h_shift_xyz_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xyt);
    su3Xsu3_h_shift_xyt_Time = std::min(su3Xsu3_h_shift_xyt_Time, timer.seconds());

    timer.reset();
    su3Xsu3_h(dev_a, dev_b, dev_d_h_shift_xyzt);
    su3Xsu3_h_shift_xyzt_Time = std::min(su3Xsu3_h_shift_xyzt_Time, timer.seconds());
  }

  int rc = 0;

  printf(HLINE);

  printf("su3Xsu3                  %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_Time);

  printf("su3Xsu3_h_no_shift                %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_no_shift_Time);
  
  printf("su3Xsu3_h_shift_x                %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_x_Time);

  printf("su3Xsu3_h_shift_y                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_y_Time);

  printf("su3Xsu3_h_shift_z                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_z_Time);

  printf("su3Xsu3_h_shift_t                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_t_Time);

  printf("su3Xsu3_h_shift_xy                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xy_Time);

  printf("su3Xsu3_h_shift_xz                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xz_Time);

  printf("su3Xsu3_h_shift_xt                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xt_Time);
          
  printf("su3Xsu3_h_shift_xyz                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xyz_Time);

  printf("su3Xsu3_h_shift_xyt                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xyt_Time);

  printf("su3Xsu3_h_shift_xyzt                %11.4f GB/s\n",
          1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / su3Xsu3_h_shift_xyzt_Time);

  printf(HLINE);

  return rc;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos 4D GaugeField (mu static, SUN as Kokkos::Array) MDRangePolicy Halo STREAM Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  StreamIndex stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_benchmark<4,3>(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}