source ${SRCDIR}/compilation/nvidia_a100/load_modules.sh 

cmake \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ARCH_ZEN2=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ${SRCDIR}
