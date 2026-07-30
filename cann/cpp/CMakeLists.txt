cmake_minimum_required(VERSION 3.10)
project(kzzk_cv CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)

# ────────────────────────────────────────────────────────────────
# 1. Ascend 路径探测
# ────────────────────────────────────────────────────────────────
set(ASCEND_SEARCH_PATHS
    $ENV{ASCEND_HOME}
    $ENV{INSTALL_DIR}/runtime
    $ENV{ASCEND_TOOLKIT_PATH}
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend/nnrt/latest
    /home/HwHiAiUser/Ascend/ascend-toolkit/latest
    /opt/Ascend/ascend-toolkit/latest
)

set(ASCEND_ARCH_SUBDIRS
    "aarch64-linux"
    "x86_64-linux"
    "arm64-linux"
    ""
)

set(ASCEND_FOUND FALSE)
set(ASCEND_ROOT "")
set(ASCEND_ARCH_SUB "")

foreach(p ${ASCEND_SEARCH_PATHS})
    if(NOT p)
        continue()
    endif()
    foreach(sub ${ASCEND_ARCH_SUBDIRS})
        if(sub)
            set(header_path "${p}/${sub}/acllib/include/acl/acl.h")
        else()
            set(header_path "${p}/acllib/include/acl/acl.h")
        endif()
        if(EXISTS "${header_path}")
            set(ASCEND_FOUND TRUE)
            set(ASCEND_ROOT "${p}")
            set(ASCEND_ARCH_SUB "${sub}")
            message(STATUS "Found Ascend at: ${ASCEND_ROOT} (arch: ${ASCEND_ARCH_SUB})")
            break()
        endif()
        if(sub)
            set(header_path2 "${p}/${sub}/include/acl/acl.h")
        else()
            set(header_path2 "${p}/include/acl/acl.h")
        endif()
        if(EXISTS "${header_path2}")
            set(ASCEND_FOUND TRUE)
            set(ASCEND_ROOT "${p}")
            set(ASCEND_ARCH_SUB "${sub}")
            message(STATUS "Found Ascend at: ${ASCEND_ROOT} (arch: ${ASCEND_ARCH_SUB})")
            break()
        endif()
    endforeach()
    if(ASCEND_FOUND)
        break()
    endif()
endforeach()

if(NOT ASCEND_FOUND)
    message(FATAL_ERROR
        "Cannot find Ascend toolkit!\n"
        "  Try setting one of these environment variables:\n"
        "    - ASCEND_HOME       (e.g. /usr/local/Ascend/ascend-toolkit/latest)\n"
        "    - INSTALL_DIR       (e.g. /usr/local/Ascend/ascend-toolkit/latest/runtime)\n"
        "  Or install Ascend toolkit to a standard location."
    )
endif()

if(ASCEND_ARCH_SUB)
    if(EXISTS "${ASCEND_ROOT}/${ASCEND_ARCH_SUB}/acllib/include/acl/acl.h")
        set(ASCEND_INCLUDE_DIRS "${ASCEND_ROOT}/${ASCEND_ARCH_SUB}/acllib/include")
    else()
        set(ASCEND_INCLUDE_DIRS "${ASCEND_ROOT}/${ASCEND_ARCH_SUB}/include")
    endif()
else()
    if(EXISTS "${ASCEND_ROOT}/acllib/include/acl/acl.h")
        set(ASCEND_INCLUDE_DIRS "${ASCEND_ROOT}/acllib/include")
    else()
        set(ASCEND_INCLUDE_DIRS "${ASCEND_ROOT}/include")
    endif()
endif()

if(ASCEND_ARCH_SUB)
    set(ascend_lib_base "${ASCEND_ROOT}/${ASCEND_ARCH_SUB}")
else()
    set(ascend_lib_base "${ASCEND_ROOT}")
endif()

set(ASCEND_LIB_SEARCH_PATHS
    "${ascend_lib_base}/acllib/lib64"
    "${ascend_lib_base}/runtime/lib64"
    "${ascend_lib_base}/lib64"
    "${ascend_lib_base}/driver/lib64"
    "${ascend_lib_base}/acllib/lib64/stub"
    "${ascend_lib_base}/runtime/lib64/stub"
    "${ascend_lib_base}/lib64/stub"
)

set(ASCEND_LIB_DIR "")
foreach(p ${ASCEND_LIB_SEARCH_PATHS})
    if(EXISTS "${p}/libascendcl.so" OR EXISTS "${p}/libascendcl.so.1" OR
       EXISTS "${p}/libacl_dvpp.so" OR EXISTS "${p}/libacl_dvpp.so.1")
        set(ASCEND_LIB_DIR "${p}")
        message(STATUS "Found Ascend lib at: ${ASCEND_LIB_DIR}")
        break()
    endif()
endforeach()

if(NOT ASCEND_LIB_DIR)
    foreach(p ${ASCEND_LIB_SEARCH_PATHS})
        if(IS_DIRECTORY "${p}")
            set(ASCEND_LIB_DIR "${p}")
            message(STATUS "Using Ascend lib dir (fallback): ${ASCEND_LIB_DIR}")
            break()
        endif()
    endforeach()
endif()

if(NOT ASCEND_LIB_DIR)
    message(FATAL_ERROR "Cannot find ascend lib directory under ${ASCEND_ROOT}")
endif()

message(STATUS "Ascend include: ${ASCEND_INCLUDE_DIRS}")
message(STATUS "Ascend lib: ${ASCEND_LIB_DIR}")

# ────────────────────────────────────────────────────────────────
# 2. OpenCV 探测
# ────────────────────────────────────────────────────────────────
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/../third_party)
set(OPENCV_THIRD_PARTY ${THIRD_PARTY_DIR}/opencv)

set(OpenCV_LIBS "")
set(OPENCV_LIB_DIR "")
set(OpenCV_INCLUDE_DIRS "")

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND AND OpenCV_LIBS)
    message(STATUS "Using system OpenCV (find_package): ${OpenCV_VERSION}")
    set(OPENCV_LIB_DIR "${OpenCV_DIR}/lib" CACHE PATH "OpenCV lib dir from find_package")
else()
    set(SYSTEM_OPENCV_SEARCH_PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
    )
    set(SYSTEM_OPENCV_INCLUDE_SEARCH
        /usr/include/opencv4
        /usr/include
        /usr/local/include/opencv4
        /usr/local/include
    )

    set(_sys_inc "")
    foreach(p ${SYSTEM_OPENCV_INCLUDE_SEARCH})
        if(EXISTS "${p}/opencv2/opencv.hpp")
            set(_sys_inc "${p}")
            message(STATUS "Found system OpenCV headers at: ${_sys_inc}")
            break()
        endif()
    endforeach()

    set(_sys_libs "")
    set(_sys_lib_dir "")
    if(_sys_inc)
        foreach(p ${SYSTEM_OPENCV_SEARCH_PATHS})
            set(_found_all TRUE)
            foreach(mod core imgproc imgcodecs)
                set(_cand "${p}/libopencv_${mod}.so")
                if(NOT EXISTS "${_cand}")
                    set(_found_all FALSE)
                    break()
                endif()
            endforeach()
            if(_found_all)
                set(_sys_lib_dir "${p}")
                foreach(mod core imgproc imgcodecs)
                    list(APPEND _sys_libs "${p}/libopencv_${mod}.so")
                endforeach()
                message(STATUS "Found system OpenCV libs at: ${_sys_lib_dir}")
                break()
            endif()
        endforeach()
    endif()

    if(_sys_inc AND _sys_libs)
        set(OpenCV_INCLUDE_DIRS "${_sys_inc}")
        set(OPENCV_LIB_DIR "${_sys_lib_dir}")
        set(OpenCV_LIBS "${_sys_libs}")
        message(STATUS "Using system OpenCV from standard paths")
    else()
        message(STATUS "System OpenCV not found, using third-party OpenCV")
        if(EXISTS "${OPENCV_THIRD_PARTY}/include/opencv4/opencv2/opencv.hpp")
            set(OpenCV_INCLUDE_DIRS "${OPENCV_THIRD_PARTY}/include/opencv4")
        else()
            set(OpenCV_INCLUDE_DIRS "${OPENCV_THIRD_PARTY}/include")
        endif()

        if(IS_DIRECTORY "${OPENCV_THIRD_PARTY}/lib64")
            set(OPENCV_LIB_DIR "${OPENCV_THIRD_PARTY}/lib64")
        else()
            set(OPENCV_LIB_DIR "${OPENCV_THIRD_PARTY}/lib")
        endif()

        foreach(mod core imgproc imgcodecs)
            set(_candidate "${OPENCV_LIB_DIR}/libopencv_${mod}.so")
            if(EXISTS "${_candidate}")
                list(APPEND OpenCV_LIBS "${_candidate}")
            else()
                file(GLOB _ver "${OPENCV_LIB_DIR}/libopencv_${mod}.so.*")
                if(_ver)
                    list(GET _ver 0 _first)
                    list(APPEND OpenCV_LIBS "${_first}")
                endif()
            endif()
        endforeach()
    endif()
endif()

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV lib dir: ${OPENCV_LIB_DIR}")

# ────────────────────────────────────────────────────────────────
# 3. 系统图像库 (libtiff, libjpeg, libpng, zlib)
# ────────────────────────────────────────────────────────────────
set(IMG_LIB_SEARCH_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    ${OPENCV_LIB_DIR}
)

function(find_system_lib OUTPUT_VAR LIB_NAME)
    set(_found "")
    foreach(p ${IMG_LIB_SEARCH_PATHS})
        set(_cand "${p}/lib${LIB_NAME}.so")
        if(EXISTS "${_cand}")
            set(_found "${_cand}")
            break()
        endif()
        file(GLOB _ver_lib "${p}/lib${LIB_NAME}.so.*")
        if(_ver_lib)
            list(GET _ver_lib 0 _found)
            break()
        endif()
    endforeach()
    if(_found)
        message(STATUS "Found lib${LIB_NAME}: ${_found}")
        set(${OUTPUT_VAR} "${_found}" PARENT_SCOPE)
    else()
        message(STATUS "lib${LIB_NAME} not found, using linker name")
        set(${OUTPUT_VAR} "${LIB_NAME}" PARENT_SCOPE)
    endif()
endfunction()

find_system_lib(JPEG_LIB jpeg)
find_system_lib(TIFF_LIB tiff)
find_system_lib(PNG_LIB png)
find_system_lib(ZLIB_LIB z)

# ────────────────────────────────────────────────────────────────
# 4. 公共头文件路径
# ────────────────────────────────────────────────────────────────
set(KZZK_CV_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/inc/models
    ${CMAKE_SOURCE_DIR}/inc/models/detection
    ${CMAKE_SOURCE_DIR}/inc/models/classification
    ${CMAKE_SOURCE_DIR}/inc/models/segmentation
    ${CMAKE_SOURCE_DIR}/utils
    ${ASCEND_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${ASCEND_LIB_DIR}
    ${OPENCV_LIB_DIR}
)

# ────────────────────────────────────────────────────────────────
# 5. 共享库 libkzzk_cv.so
# ────────────────────────────────────────────────────────────────
set(KZZK_CV_LIB_SOURCES
    src/base_model.cpp
    src/kzzk_cv.cpp
    src/models/detection/yolov3.cpp
    src/models/detection/yolov4.cpp
    src/models/detection/vgg_ssd.cpp
    src/models/detection/rt_detr.cpp
    src/models/classification/resnet50.cpp
    src/models/classification/mobilenetv1.cpp
    src/models/classification/vgg16.cpp
    src/models/segmentation/deeplabv3.cpp
)

add_library(kzzk_cv_lib SHARED ${KZZK_CV_LIB_SOURCES})

target_include_directories(kzzk_cv_lib
    PUBLIC ${KZZK_CV_INCLUDE_DIRS}
)

target_link_libraries(kzzk_cv_lib
    ascendcl
    acl_dvpp
    ${OpenCV_LIBS}
    ${JPEG_LIB}
    ${TIFF_LIB}
    ${PNG_LIB}
    ${ZLIB_LIB}
    stdc++
    dl
    pthread
)

# 运行时库搜索路径：包含自身目录、OpenCV、Ascend 及常见系统路径
# 必须把 OPENCV_LIB_DIR 和 ASCEND_LIB_DIR 写进来，否则可执行文件运行时
# 找不到 libopencv_core.so.413 / libascendcl.so 等依赖
set(KZZK_CV_RPATH
    "\$ORIGIN:\$ORIGIN/../lib:${OPENCV_LIB_DIR}:${ASCEND_LIB_DIR}:/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/lib64:/usr/local/lib64:/usr/local/lib"
)

# 设置 .so 版本号和 RPATH
set_target_properties(kzzk_cv_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME kzzk_cv
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${KZZK_CV_RPATH}"
    LINK_FLAGS "-Wl,--unresolved-symbols=ignore-in-shared-libs"
)

# ────────────────────────────────────────────────────────────────
# 6. 命令行工具 kzzk_cv (链接共享库)
# ────────────────────────────────────────────────────────────────
add_executable(kzzk_cv src/main.cpp)

target_include_directories(kzzk_cv
    PRIVATE ${CMAKE_SOURCE_DIR}/inc
)

target_link_libraries(kzzk_cv kzzk_cv_lib)

# RPATH：运行时从可执行目录找 libkzzk_cv.so 及其 OpenCV/Ascend 依赖
set_target_properties(kzzk_cv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${KZZK_CV_RPATH}"
)

# ────────────────────────────────────────────────────────────────
# 7. 示例可执行文件 test_cv (调用 libkzzk_cv.so，演示 kzzk_cv 接口)
# ────────────────────────────────────────────────────────────────
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/../examples)

add_executable(test_cv ${EXAMPLES_DIR}/test_cv.cpp)

target_include_directories(test_cv
    PRIVATE ${CMAKE_SOURCE_DIR}/inc
)

target_link_libraries(test_cv kzzk_cv_lib)

# RPATH：让 test_cv 能找到 libkzzk_cv.so 及其 OpenCV/Ascend 依赖
set_target_properties(test_cv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${KZZK_CV_RPATH}"
)

# ────────────────────────────────────────────────────────────────
# 8. 安装规则
# ────────────────────────────────────────────────────────────────
install(TARGETS kzzk_cv_lib
    LIBRARY DESTINATION lib
)

install(TARGETS kzzk_cv test_cv
    RUNTIME DESTINATION bin
)

# 安装公共头文件
install(FILES
    ${CMAKE_SOURCE_DIR}/inc/kzzk_cv.h
    ${CMAKE_SOURCE_DIR}/inc/types.h
    ${CMAKE_SOURCE_DIR}/inc/base_model.h
    DESTINATION include/kzzk_cv
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/inc/models/
    DESTINATION include/kzzk_cv/models
    FILES_MATCHING PATTERN "*.h"
)

install(FILES
    ${CMAKE_SOURCE_DIR}/utils/coco_labels.h
    ${CMAKE_SOURCE_DIR}/utils/voc_labels.h
    DESTINATION include/kzzk_cv/utils
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Ascend root: ${ASCEND_ROOT}")
message(STATUS "Output lib:  ${CMAKE_BINARY_DIR}/lib/libkzzk_cv.so.1")
message(STATUS "Output exe:  ${CMAKE_BINARY_DIR}/bin/kzzk_cv")
message(STATUS "Example exe: ${CMAKE_BINARY_DIR}/bin/test_cv")
