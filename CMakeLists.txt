cmake_minimum_required(VERSION 3.16)
project(demo)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
igl_include(imgui)
# igl_include(opengl)
igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
# igl_include(copyleft core)
#igl_include(copyleft cork)
#igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

# Find all cpp files under the src directory
file(GLOB SRC_FILES src/*.cpp)

# Executables will be placed in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Generate an executable for each cpp file
foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
    # Replace spaces with underscores to avoid invalid target names
    string(REPLACE " " "_" EXE_NAME ${EXE_NAME})
    add_executable(${EXE_NAME} ${SRC_FILE})
    target_link_libraries(${EXE_NAME} PUBLIC igl::glfw igl::imgui igl::stb)
endforeach()

# # Only build src/demo.cpp as an executable
# add_executable(demo src/demo.cpp)
# target_link_libraries(demo PUBLIC igl::glfw igl::imgui igl::stb)

# Link igl (and the glfw module) to your project
# target_link_libraries(${PROJECT_NAME} PUBLIC 
#   igl::glfw
#   ## Other modules you could link to
#   # igl::embree
#   # igl::imgui
#   # igl::opengl
#   # igl::stb
#   # igl::predicates
#   # igl::xml
#   # igl_copyleft::cgal
#   # igl_copyleft::comiso
#   # igl_copyleft::core
#   # igl_copyleft::cork
#   # igl_copyleft::tetgen
#   # igl_restricted::matlab
#   # igl_restricted::mosek
#   # igl_restricted::triangle
#   )
