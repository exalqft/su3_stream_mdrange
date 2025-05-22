cmake_minimum_required(VERSION 3.21.0)
project(su3-stream-mdrange VERSION 1.0.0 LANGUAGES CXX)

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++20
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

add_executable(su3xsu3-stream-mdrange-static su3xsu3-stream-mdrange-static.cpp)
target_link_libraries(su3xsu3-stream-mdrange-static Kokkos::kokkos)

add_executable(su3xsu3-stream-mdrange-static-mu su3xsu3-stream-mdrange-static-mu.cpp)
target_link_libraries(su3xsu3-stream-mdrange-static-mu Kokkos::kokkos)

add_executable(su3xsu3-stream-mdrange-static-mu-array-SUN su3xsu3-stream-mdrange-static-mu-array-SUN.cpp)
target_link_libraries(su3xsu3-stream-mdrange-static-mu-array-SUN Kokkos::kokkos)

add_executable(su3xsu3-stream-mdrange-static-mu-reftype su3xsu3-stream-mdrange-static-mu-reftype.cpp)
target_link_libraries(su3xsu3-stream-mdrange-static-mu-reftype Kokkos::kokkos)

add_executable(su3xsu3-stream-1drange-static-mu-reftype su3xsu3-stream-1drange-static-mu-reftype.cpp)
target_link_libraries(su3xsu3-stream-1drange-static-mu-reftype Kokkos::kokkos)
