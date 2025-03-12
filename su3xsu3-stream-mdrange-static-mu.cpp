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

using real_t = double;
using val_t = Kokkos::complex<real_t>;
constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);
constexpr val_t dinit(1.4, 0.4);

//using val_t = double;
//constexpr val_t ainit(1.0);
//constexpr val_t binit(1.1);
//constexpr val_t cinit(1.3);

#define HLINE "-------------------------------------------------------------\n"

template <int Nd, int Nc>
using GaugeField =
    Kokkos::View<val_t****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using SUNField =
    Kokkos::View<val_t****[Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;


#if defined(KOKKOS_ENABLE_CUDA)
template <int Nd, int Nc>
using constGaugeField =
    Kokkos::View<const val_t ****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <int Nc>
using constSUNField =
    Kokkos::View<const val_t****[Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <int Nd, int Nc>
using constGaugeField =
    Kokkos::View<const val_t ****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <int Nc>
using constSUNField =
    Kokkos::View<const val_t****[Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
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
              V(i,j,k,l,mu,c1,c2) = init;
            }
          }
        }
      }
    );
    Kokkos::fence();
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
            V(i,j,k,l,c1,c2) = init;
          }
        }
      }
    );
    Kokkos::fence();
  }

  SUNField<Nc> view;
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
void perform_matmul(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
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
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              a.view(i,j,k,l,mu,c1,c2) = b.view(i,j,k,l,mu,c1,0) * c.view(i,j,k,l,mu,0,c2);
              #pragma unroll
              for(int ci = 1; ci < Nc; ++ci){
                a.view(i,j,k,l,mu,c1,c2) += b.view(i,j,k,l,mu,c1,ci) * c.view(i,j,k,l,mu,ci,c2);
              }
            }
          }
        }
      });

  Kokkos::fence();
}

template <int Nd, int Nc>
void perform_matmul_tmp(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
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
        val_t tmp;
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              tmp = b.view(i,j,k,l,mu,c1,0) * c.view(i,j,k,l,mu,0,c2);
              #pragma unroll
              for(int ci = 1; ci < Nc; ++ci){
                tmp += b.view(i,j,k,l,mu,c1,ci) * c.view(i,j,k,l,mu,ci,c2);
              }
              a.view(i,j,k,l,mu,c1,c2) = tmp;
            }
          }
        }
      });

  Kokkos::fence();
}

template <int Nd, int Nc>
void perform_conj_matmul_tmp(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
                             const deviceGaugeField<Nd,Nc> c) {
  constexpr auto rank = a.view.rank_dynamic();
  const auto stream_array_size = a.view.extent(0);
  const auto tiling = get_tiling(a.view);
  Kokkos::parallel_for(
      "conjsu3xsu3", 
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        val_t tmp;
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              tmp = conj(b.view(i,j,k,l,mu,0,c1)) * c.view(i,j,k,l,mu,0,c2);
              #pragma unroll
              for(int ci = 1; ci < Nc; ++ci){
                tmp += conj(b.view(i,j,k,l,mu,ci,c1)) * c.view(i,j,k,l,mu,ci,c2);
              }
              a.view(i,j,k,l,mu,c1,c2) = tmp;
            }
          }
        }
      });

  Kokkos::fence();
}

template <int Nd, int Nc>
void perform_matmul_inter(const deviceGaugeField<Nd,Nc> a, const deviceGaugeField<Nd,Nc> b,
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
        val_t tmp;
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int ci = 0; ci < Nc; ++ci){
              tmp = b.view(i,j,k,l,mu,c1,ci);
              #pragma unroll
              for(int c2 = 0; c2 < Nc; ++c2){
                a.view(i,j,k,l,mu,c1,c2) += tmp * c.view(i,j,k,l,mu,ci,c2);
              }
            }
          }
        }
      });

  Kokkos::fence();
}

template <int Nd, int Nc>    
void perform_halfstaple(const deviceSUNField<Nc> d, const deviceGaugeField<Nd,Nc> g,    
                        const int mu, const int nu)    
{    
  assert(mu < Nd && nu < Nd);    
  constexpr auto rank = d.view.rank_dynamic();    
  const auto stream_array_size = d.view.extent(0);    
  const auto tiling = get_tiling(d.view);    
  Kokkos::parallel_for(    
    "suN_halfstaple",     
    Policy<rank>(make_repeated_sequence<rank>(0),     
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size},     
                 tiling),    
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)    
    {    
      const int ipmu = mu == 0 ? (i + 1) % stream_array_size : i;    
      const int jpmu = mu == 1 ? (j + 1) % stream_array_size : j;    
      const int kpmu = mu == 2 ? (k + 1) % stream_array_size : k;    
      const int lpmu = mu == 3 ? (l + 1) % stream_array_size : l;    
      val_t tmp;    
      #pragma unroll    
      for(int c1 = 0; c1 < Nc; ++c1){    
        #pragma unroll    
        for(int c2 = 0; c2 < Nc; ++c2){    
          tmp = g.view(i,j,k,l,mu,c1,0) * g.view(ipmu,jpmu,kpmu,lpmu,nu,0,c2);    
          #pragma unroll    
          for(int ci = 1; ci < Nc; ++ci){    
            tmp += g.view(i,j,k,l,mu,c1,ci) * g.view(ipmu,jpmu,kpmu,lpmu,nu,ci,c2);    
          }    
          d.view(i,j,k,l,c1,c2) = tmp;    
        }    
      }    
    });    
  Kokkos::fence();    
}    

template<int mu, int shift, typename SizeType>
constexpr
KOKKOS_FORCEINLINE_FUNCTION
StreamIndex
shift_index(const StreamIndex & i, const SizeType & stream_array_size)
{
  if constexpr ( mu == shift ){
    return (i + 1) % stream_array_size;
  } else {
    return i;
  }
}

template <int Nd, int Nc, int mu, int nu, typename SizeType>
KOKKOS_FORCEINLINE_FUNCTION
void
plaq_kernel(Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> & lmu,
            Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> & lnu,
            const constGaugeField<Nd,Nc> & g,
            const StreamIndex & i, const StreamIndex & j, const StreamIndex & k, const StreamIndex & l,
            val_t & lres, val_t & tmu, val_t & tnu, const SizeType & stream_array_size)
{
  const StreamIndex ipmu = shift_index<0,mu>(i, stream_array_size);
  const StreamIndex jpmu = shift_index<1,mu>(j, stream_array_size);
  const StreamIndex kpmu = shift_index<2,mu>(k, stream_array_size);
  const StreamIndex lpmu = shift_index<3,mu>(l, stream_array_size);
  const StreamIndex ipnu = shift_index<0,nu>(i, stream_array_size);
  const StreamIndex jpnu = shift_index<1,nu>(j, stream_array_size);
  const StreamIndex kpnu = shift_index<2,nu>(k, stream_array_size);
  const StreamIndex lpnu = shift_index<3,nu>(l, stream_array_size);

  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      tmu = g(i,j,k,l,mu,c1,0) * g(ipmu,jpmu,kpmu,lpmu,nu,0,c2);
      tnu = g(i,j,k,l,nu,c1,0) * g(ipnu,jpnu,kpnu,lpnu,mu,0,c2);
      #pragma unroll
      for(int ci = 1; ci < Nc; ++ci){
        tmu += g(i,j,k,l,mu,c1,ci) * g(ipmu,jpmu,kpmu,lpmu,nu,ci,c2);
        tnu += g(i,j,k,l,nu,c1,ci) * g(ipnu,jpnu,kpnu,lpnu,mu,ci,c2);
      }
      lmu[c1][c2] = tmu;
      lnu[c1][c2] = tnu;
    }
  }

  #pragma unroll
  for(int c = 0; c < Nc; ++c){
    #pragma unroll
    for(int ci = 0; ci < Nc; ++ci){
      lres += lmu[c][ci] * conj(lnu[c][ci]);
    }
  }
}

template <int Nd, int Nc>
val_t perform_plaquette_kernel(const deviceGaugeField<Nd,Nc> g_in)
{
  assert(mu < Nd && nu < Nd);
  constexpr auto rank = g_in.view.rank_dynamic();
  const auto stream_array_size = g_in.view.extent(0);
  const auto tiling = get_tiling(g_in.view);

  const constGaugeField<Nd,Nc> g(g_in.view); 
  
  val_t res = 0.0;

  Kokkos::parallel_reduce(
    "suN_plaquette_kernel", 
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                  val_t & lres)
    {
      Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> lmu, lnu;

      val_t tmu, tnu;

      if (Nd > 1) plaq_kernel<Nd,Nc,0,1>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);
      if (Nd > 2) plaq_kernel<Nd,Nc,0,2>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);
      if (Nd > 3) plaq_kernel<Nd,Nc,0,3>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);
      if (Nd > 2) plaq_kernel<Nd,Nc,1,2>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);
      if (Nd > 3) plaq_kernel<Nd,Nc,1,3>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);
      if (Nd > 3) plaq_kernel<Nd,Nc,2,3>(lmu, lnu, g, i, j, k, l, lres, tmu, tnu, stream_array_size);

    }, Kokkos::Sum<val_t>(res) );
  Kokkos::fence();
  return res;
}

template <int Nd, int Nc>
val_t perform_plaquette(const deviceGaugeField<Nd,Nc> g_in)
{
  assert(mu < Nd && nu < Nd);
  constexpr auto rank = g_in.view.rank_dynamic();
  const auto stream_array_size = g_in.view.extent(0);
  const auto tiling = get_tiling(g_in.view);

  const constGaugeField<Nd,Nc> g(g_in.view); 
  
  val_t res = 0.0;

  Kokkos::parallel_reduce(
    "suN_plaquette", 
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                  val_t & lres)
    {
      Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> lmu, lnu;

      val_t tmu, tnu;

      #pragma unroll
      for(int mu = 0; mu < Nd; ++mu){
        #pragma unroll
        for(int nu = 0; nu < Nd; ++nu){
          // unrolling only works well with constant-value loop limits
          if( nu > mu ){
            const StreamIndex ipmu = mu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpmu = mu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpmu = mu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpmu = mu == 3 ? (l + 1) % stream_array_size : l;
            const StreamIndex ipnu = nu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpnu = nu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpnu = nu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpnu = nu == 3 ? (l + 1) % stream_array_size : l;
            #pragma unroll
            for(int c1 = 0; c1 < Nc; ++c1){
              #pragma unroll
              for(int c2 = 0; c2 < Nc; ++c2){
                tmu = g(i,j,k,l,mu,c1,0) * g(ipmu,jpmu,kpmu,lpmu,nu,0,c2);
                tnu = g(i,j,k,l,nu,c1,0) * g(ipnu,jpnu,kpnu,lpnu,mu,0,c2);
                #pragma unroll
                for(int ci = 1; ci < Nc; ++ci){
                  tmu += g(i,j,k,l,mu,c1,ci) * g(ipmu,jpmu,kpmu,lpmu,nu,ci,c2);
                  tnu += g(i,j,k,l,nu,c1,ci) * g(ipnu,jpnu,kpnu,lpnu,mu,ci,c2);
                }
                lmu[c1][c2] = tmu;
                lnu[c1][c2] = tnu;
              }
            }
            #pragma unroll
            for(int c = 0; c < Nc; ++c){
              #pragma unroll
              for(int ci = 0; ci < Nc; ++ci){
                lres += lmu[c][ci] * conj(lnu[c][ci]);
              }
            }
          }
        }
      }
    }, Kokkos::Sum<val_t>(res) );
  Kokkos::fence();
  return res;
}

template <int Nd, int Nc>
void perform_plaquette_notrace(const deviceSUNField<Nc> plaq_out, const deviceGaugeField<Nd,Nc> g_in)
{
  assert(mu < Nd && nu < Nd);
  constexpr auto rank = g_in.view.rank_dynamic();
  const auto stream_array_size = g_in.view.extent(0);
  const auto tiling = get_tiling(g_in.view);

  const constGaugeField<Nd,Nc> g(g_in.view); 
  
  Kokkos::parallel_for(
    "suN_plaquette", 
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
    {
      Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> lmu, lnu;

      val_t tmu, tnu;
      
      #pragma unroll
      for(int c = 0; c < Nc; ++c){
        plaq_out.view(i,j,k,l,c,c) = 0;
      }

      #pragma unroll
      for(int mu = 0; mu < Nd; ++mu){
        #pragma unroll
        for(int nu = 0; nu < Nd; ++nu){
          // unrolling only works well with constant-value loop limits
          if( nu > mu ){
            const StreamIndex ipmu = mu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpmu = mu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpmu = mu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpmu = mu == 3 ? (l + 1) % stream_array_size : l;
            const StreamIndex ipnu = nu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpnu = nu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpnu = nu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpnu = nu == 3 ? (l + 1) % stream_array_size : l;
            #pragma unroll
            for(int c1 = 0; c1 < Nc; ++c1){
              #pragma unroll
              for(int c2 = 0; c2 < Nc; ++c2){
                tmu = g(i,j,k,l,mu,c1,0) * g(ipmu,jpmu,kpmu,lpmu,nu,0,c2);
                tnu = g(i,j,k,l,nu,c1,0) * g(ipnu,jpnu,kpnu,lpnu,mu,0,c2);
                #pragma unroll
                for(int ci = 1; ci < Nc; ++ci){
                  tmu += g(i,j,k,l,mu,c1,ci) * g(ipmu,jpmu,kpmu,lpmu,nu,ci,c2);
                  tnu += g(i,j,k,l,nu,c1,ci) * g(ipnu,jpnu,kpnu,lpnu,mu,ci,c2);
                }
                lmu[c1][c2] = tmu;
                lnu[c1][c2] = tnu;
              }
            }
            #pragma unroll
            for(int c = 0; c < Nc; ++c){
              tmu = lmu[c][0] * conj(lnu[c][0]);
              #pragma unroll
              for(int ci = 1; ci < Nc; ++ci){
                tmu += lmu[c][ci] * conj(lnu[c][ci]);
              }
              // we sum up all the plaquettes since we are only interested
              // in the trace -> we do only the diagonal
              plaq_out.view(i,j,k,l,c,c) += tmu;
            }
          }
        }
      }
    });
  Kokkos::fence();
}

template <int Nc>
val_t perform_sun_trace(const deviceSUNField<Nc> plaq_in)
{
  constexpr auto rank = plaq_in.view.rank_dynamic();
  const auto stream_array_size = plaq_in.view.extent(0);
  const auto tiling = get_tiling(plaq_in.view);

  const constSUNField<Nc> plaq(plaq_in.view); 
  
  val_t res = 0.0;

  Kokkos::parallel_reduce(
    "suN_trace", 
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                  val_t & lres)
    {
      #pragma unroll
      for(int c = 0; c < Nc; ++c){
        lres += plaq(i,j,k,l,c,c);
      }
    }, Kokkos::Sum<val_t>(res) );
  Kokkos::fence();
  return res;
}

// int perform_validation(StreamHostArray &a, StreamHostArray &b,
//                        StreamHostArray &c, const StreamIndex arraySize,
//                        const val_t scalar) {
//   val_t ai = ainit;
//   val_t bi = binit;
//   val_t ci = cinit;
// 
//   for (StreamIndex i = 0; i < STREAM_NTIMES; ++i) {
//     ci = ai;
//     bi = scalar * ci;
//     ci = ai + bi;
//     ai = bi + scalar * ci;
//   };
// 
//   std::cout << "ai: " << ai << "\n";
//   std::cout << "a(0,0,0,0): " << a(0,0,0,0) << "\n";
//   std::cout << "bi: " << bi << "\n";
//   std::cout << "b(0,0,0,0): " << b(0,0,0,0) << "\n";
//   std::cout << "ci: " << ci << "\n";
//   std::cout << "c(0,0,0,0): " << c(0,0,0,0) << "\n";
//  
//   const double nelem = (double)arraySize*arraySize*arraySize*arraySize; 
//   const double epsilon = 2*4*STREAM_NTIMES*std::numeric_limits<val_t>::epsilon();
// 
//   double aError = 0.0;
//   double bError = 0.0;
//   double cError = 0.0;
// 
//   #pragma omp parallel reduction(+:aError,bError,cError)
//   {
//     double err = 0.0;
//     #pragma omp for collapse(2)
//     for (StreamIndex i = 0; i < arraySize; ++i) {
//       for (StreamIndex j = 0; j < arraySize; ++j) {
//         for (StreamIndex k = 0; k < arraySize; ++k) {
//           for (StreamIndex l = 0; l < arraySize; ++l) {
//             err = std::abs(a(i,j,k,l) - ai);
//             if( err > epsilon ){
//               //std::cout << "aError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               aError += err;
//             }
//             err = std::abs(b(i,j,k,l) - bi);
//             if( err > epsilon ){
//               //std::cout << "bError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               bError += err;
//             }
//             err = std::abs(c(i,j,k,l) - ci);
//             if( err > epsilon ){
//               //std::cout << "cError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               cError += err;
//             }
//           }
//         }
//       }
//     }
//   }
// 
//   std::cout << "aError = " << aError << "\n";
//   std::cout << "bError = " << bError << "\n";
//   std::cout << "cError = " << cError << "\n";
// 
//   val_t aAvgError = aError / nelem;
//   val_t bAvgError = bError / nelem;
//   val_t cAvgError = cError / nelem;
// 
//   std::cout << "aAvgErr = " << aAvgError << "\n";
//   std::cout << "bAvgError = " << bAvgError << "\n";
//   std::cout << "cAvgError = " << cAvgError << "\n";
// 
//   int errorCount       = 0;
// 
//   if (std::abs(aAvgError / ai) > epsilon) {
//     fprintf(stderr, "Error: validation check on View a failed.\n");
//     errorCount++;
//   }
// 
//   if (std::abs(bAvgError / bi) > epsilon) {
//     fprintf(stderr, "Error: validation check on View b failed.\n");
//     errorCount++;
//   }
// 
//   if (std::abs(cAvgError / ci) > epsilon) {
//     fprintf(stderr, "Error: validation check on View c failed.\n");
//     errorCount++;
//   }
// 
//   if (errorCount == 0) {
//     printf("All solutions checked and verified.\n");
//   }
// 
//   return errorCount;
// }

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

  // WithoutInitializing to circumvent first touch bug on arm systems
  // GaugeField dev_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "a"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);
  // GaugeField dev_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);
  // GaugeField dev_c(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);

  // StreamHostArray a = Kokkos::create_mirror_view(dev_a);
  // StreamHostArray b = Kokkos::create_mirror_view(dev_b);
  // StreamHostArray c = Kokkos::create_mirror_view(dev_c);

  double matmulTime  = std::numeric_limits<double>::max();
  double matmulTmpTime  = std::numeric_limits<double>::max();
  double matmulInterTime  = std::numeric_limits<double>::max();
  double conjMatmulTime  = std::numeric_limits<double>::max();
  double halfstapleTime = std::numeric_limits<double>::max();
  double plaquetteTime = std::numeric_limits<double>::max();
  double plaquetteKernelTime = std::numeric_limits<double>::max();
  double plaquetteNotraceTime = std::numeric_limits<double>::max();
  double plaquetteTraceTime = std::numeric_limits<double>::max();
  double plaquetteNotraceTraceTime = std::numeric_limits<double>::max();

  printf("Initializing Views...\n");

  deviceGaugeField<Nd,Nc> dev_a(stream_array_size,stream_array_size,stream_array_size,stream_array_size,ainit);
  deviceGaugeField<Nd,Nc> dev_b(stream_array_size,stream_array_size,stream_array_size,stream_array_size,binit);
  deviceGaugeField<Nd,Nc> dev_c(stream_array_size,stream_array_size,stream_array_size,stream_array_size,cinit);
  deviceSUNField<Nc> dev_d(stream_array_size,stream_array_size,stream_array_size,stream_array_size,dinit);

  printf("Starting benchmarking...\n");

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    perform_matmul(dev_a, dev_b, dev_c);
    matmulTime = std::min(matmulTime, timer.seconds());
    
    timer.reset();
    perform_matmul_tmp(dev_a, dev_b, dev_c);
    matmulTmpTime = std::min(matmulTmpTime, timer.seconds());
    
    timer.reset();
    perform_matmul_inter(dev_a, dev_b, dev_c);
    matmulInterTime = std::min(matmulInterTime, timer.seconds());
  
    timer.reset();
    perform_conj_matmul_tmp(dev_a, dev_b, dev_c);
    conjMatmulTime = std::min(conjMatmulTime, timer.seconds());

    timer.reset();
    perform_halfstaple(dev_d, dev_a,2,0);
    halfstapleTime = std::min(halfstapleTime, timer.seconds());

    timer.reset();
    val_t plaq = perform_plaquette(dev_b);
    plaquetteTime = std::min(plaquetteTime, timer.seconds());
    //if (k == 2) std::cout << "Plaquette: " << plaq << "\n";
    
    timer.reset();
    plaq = perform_plaquette_kernel(dev_c);
    plaquetteKernelTime = std::min(plaquetteKernelTime, timer.seconds());
    //if (k == 2) std::cout << "Plaquette(Kernel): " << plaq << "\n";

    timer.reset();
    perform_plaquette_notrace(dev_d, dev_b);
    plaquetteNotraceTime = std::min(plaquetteNotraceTime, timer.seconds());

    timer.reset();
    perform_sun_trace(dev_d);
    plaquetteTraceTime = std::min(plaquetteTraceTime, timer.seconds());

    timer.reset();
    perform_plaquette_notrace(dev_d, dev_c);
    plaq = perform_sun_trace(dev_d);
    plaquetteNotraceTraceTime = std::min(plaquetteNotraceTraceTime, timer.seconds());
    //if (k == 2) std::cout << "Plaquette(Notrace,Trace): " << plaq << "\n";
  }

  // Kokkos::deep_copy(a, dev_a);
  // Kokkos::deep_copy(b, dev_b);
  // Kokkos::deep_copy(c, dev_c);

  // printf("Performing validation...\n");
  // int rc = perform_validation(a, b, c, stream_array_size, scalar);

  int rc = 0;

  printf(HLINE);

  printf("MatMul                  %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / matmulTime);
  
  printf("MatMulTmp               %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / matmulTmpTime);
  
  printf("MatMulInter             %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / matmulInterTime);

  printf("conjMatMul              %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * gauge_nelem / conjMatmulTime);
  
  printf("HalfStaple              %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * suN_nelem / halfstapleTime);
  
  printf("Plaquette               %11.4f GB/s %11.4e s\n",
         1.0e-09 * 1.0 * (double)sizeof(val_t) * gauge_nelem / plaquetteTime,
         plaquetteTime);

  printf("Plaquette Kernel        %11.4f GB/s %11.4e s\n",
         1.0e-09 * 1.0 * (double)sizeof(val_t) * gauge_nelem / plaquetteKernelTime,
         plaquetteKernelTime);
  
  // the Notrace plaquette kernel only writes the diagonal elements
  // of the output SUNField
  printf("Plaquette Notrace       %11.4f GB/s %11.4e s\n",
         1.0e-09 * (4.0 + 1.0/3.0) * (double)sizeof(val_t) * suN_nelem / plaquetteNotraceTime,
         plaquetteNotraceTime);

  // the Trace only reads the diagonal elements of the input SUN Field
  printf("Plaquette Trace         %11.4f GB/s %11.4e s\n",
         1.0e-09 * (1.0/3.0) * (double)sizeof(val_t) * suN_nelem / plaquetteTraceTime,
         plaquetteTraceTime);

  // NotraceTrace = "Plaquette Notrace + Plaquette Trace" -> only writes and reads the diagonal elements
  printf("Plaquette NotraceTrace  %11.4f GB/s %11.4e s\n",
         1.0e-09 * (4.0 + 2.0/3.0) * (double)sizeof(val_t) * suN_nelem / plaquetteNotraceTraceTime,
         plaquetteNotraceTraceTime);

  printf("\n"
         "Plaquette               %11.4f Gflop/s\n",
         1.0e-09 * nelem *
           6.0 *                 // six plaquettes in 4D      
         ( (2 * 9 * (18 + 4)) +  // two su3 multiplications
           (3 * (18 + 4))     +  // one su3 multiplcation (diagonal elements only)
           (3) )                 // trace accumulation (our plaquette is complex but we're interested only in the real part)
         / plaquetteTime); 

  printf("Plaquette Kernel        %11.4f Gflop/s\n",
         1.0e-09 * 6.0 * nelem * // six plaquettes in 4D
         ( (2 * 9 * (18 + 4)) +  // two su3 mults
           (3 * (18 + 4))     +  // one su3 mult (diagonal elems only)
           (3) )                 // trace accumulation (complex but we care only about the real part)
         / plaquetteKernelTime);

  printf("Plaquette Notrace       %11.4f Gflop/s\n",
         1.0e-09 * nelem *
           6.0 *                 // six plaquettes in 4D      
         ( (2 * 9 * (18 + 4)) +  // two su3 multiplications
           (3 * (18 + 4)))       // one su3 multiplcation (diagonal elements only)
         / plaquetteNotraceTime); 

  printf(HLINE);

  return rc;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos 4D GaugeField (mu static) MDRangePolicy STREAM Benchmark\n");
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
