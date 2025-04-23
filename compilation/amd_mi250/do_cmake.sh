source ${SRCDIR}/compilation/amd_mi250/load_modules.sh 

cmake \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_ZEN3=ON \
  -DKokkos_ARCH_AMD_GFX90A=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ${SRCDIR}