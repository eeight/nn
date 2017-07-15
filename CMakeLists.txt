cmake_minimum_required(VERSION 3.5)
enable_testing()

project(nn)

find_package(Armadillo REQUIRED)

include_directories(${Armadillo_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -fno-omit-frame-pointer")
set(CMAKE_CXX_STANDARD 14)

add_executable(main main.cc mnist.cc activation.cc loss.cc)
target_link_libraries(main ${ARMADILLO_LIBRARY})
