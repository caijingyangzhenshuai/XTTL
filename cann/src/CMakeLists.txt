# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(kzzk_cann)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "./lib")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)

# Find OpenCV
find_package(OpenCV QUIET)

# 如果找不到OpenCV，尝试使用third_party下的OpenCV
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found, trying third_party version...")
    set(OpenCV_DIR "${PROJECT_SOURCE_DIR}/../third_party/opencv/build")
    find_package(OpenCV REQUIRED)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found in third_party either. Please install OpenCV or place it in third_party/opencv")
    endif()
endif()

# Header path
include_directories(
    ../inc/                          # 项目头文件目录
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/  # AscendCL头文件
    ${OpenCV_INCLUDE_DIRS}           # OpenCV头文件
)

# Library path
link_directories(
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/  # AscendCL库文件
    ${OpenCV_LIB_DIR}                # OpenCV库文件
)

# 编译核心动态库
add_library(kzzk_acl_lib SHARED
    yolov4.cpp                       # YOLOV4实现
    resnet50.cpp                     # RESNET50实现
)

# 链接核心库的依赖
target_link_libraries(kzzk_acl_lib PUBLIC
    ascendcl                         # AscendCL库
    acl_dvpp                         # DVPP库
    ${OpenCV_LIBS}                   # OpenCV库
    stdc++                           # C++标准库
)

# 设置核心库的运行时库路径
set_target_properties(kzzk_acl_lib PROPERTIES
    INSTALL_RPATH "/usr/local/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 编译主可执行文件
add_executable(kzzk_cv
    main.cpp                         # 主程序
)

# 链接核心库
target_link_libraries(kzzk_cv PRIVATE kzzk_acl_lib)

# 设置可执行文件的运行时库路径
set_target_properties(kzzk_cv PROPERTIES
    INSTALL_RPATH "/usr/local/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"  # 编译期找lib/kzzk_acl_lib.so
)

# 编译ResNet50示例程序
add_executable(test_cv
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/test_cv.cpp             # ResNet50示例程序
)

# 链接核心库
target_link_libraries(test_cv PRIVATE kzzk_acl_lib)

# 设置可执行文件的运行时库路径
set_target_properties(test_cv PROPERTIES
    INSTALL_RPATH "/usr/local/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"  # 编译期找lib/kzzk_acl_lib.so
)

# Install configuration
# 安装核心库
install(TARGETS kzzk_acl_lib
    LIBRARY DESTINATION /usr/local/lib64
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# 安装可执行文件
install(TARGETS kzzk_cv
    RUNTIME DESTINATION /usr/bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Uninstall configuration
if(NOT TARGET uninstall)
    # Generate uninstall script directly in CMakeLists.txt
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        "# cmake_uninstall.cmake\n"
        "\n"
        "# Find installed files\n"
        "if(NOT EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\")\n"
        "    message(FATAL_ERROR \"Cannot find install manifest: ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\")\n"
        "endif()\n"
        "\n"
        "file(READ \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\" files)\n"
        "string(REGEX REPLACE \"\\n\" \";\" files \"\${files}\")\n"
        "\n"
        "foreach(file \${files})\n"
        "    message(STATUS \"Uninstalling: \${file}\")\n"
        "    if(EXISTS \"\${file}\")\n"
        "        execute_process(\n"
        "            COMMAND ${CMAKE_COMMAND} -E remove \"\${file}\"\n"
        "            RESULT_VARIABLE result\n"
        "            ERROR_VARIABLE output\n"
        "        )\n"
        "        if(NOT result EQUAL 0)\n"
        "            message(FATAL_ERROR \"Failed to remove \${file}: \${output}\")\n"
        "        endif()\n"
        "    else()\n"
        "        message(STATUS \"File not found: \${file}\")\n"
        "    endif()\n"
        "endforeach()\n"
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling kzzk_cv..."
    )
endif()


