cmake_minimum_required(VERSION 3.13.4)

find_package(Slicer COMPONENTS ConfigurePrerequisites)

project(DDTracking)

set(EXTENSION_HOMEPAGE "https://github.com/yishengpoxiao/DDTracking")
set(EXTENSION_CATEGORY "Diffusion MRI")
set(EXTENSION_CONTRIBUTORS "DDTracking contributors")
set(EXTENSION_DESCRIPTION "Run DDTracking tractography and FA registration from 3D Slicer.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(DDTracking)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
