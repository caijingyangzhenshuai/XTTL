cmake_minimum_required(VERSION 3.16)

project(kzzk_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/../third_party/json/include
)

# 查找依赖
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# 源代码文件
set(SOURCES
    src/kzzk_llm.cpp
)

# 创建静态库
add_library(kzzk_llm_static STATIC ${SOURCES})

# 链接依赖（使用系统 curl）
target_link_libraries(kzzk_llm_static
    PRIVATE
    CURL::libcurl
    Threads::Threads
)

# 创建命令行工具
add_executable(kzzk_llm src/main.cpp)

target_link_libraries(kzzk_llm
    PRIVATE
    kzzk_llm_static
    CURL::libcurl
    Threads::Threads
)

# 添加示例程序
add_executable(test_llm ../examples/test_llm.cpp)

target_link_libraries(test_llm
    PRIVATE
    kzzk_llm_static
    CURL::libcurl
    Threads::Threads
)

# 安装目标
install(TARGETS kzzk_llm kzzk_llm_static
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装头文件
install(DIRECTORY inc/ DESTINATION include)

# 打印配置信息
message(STATUS "CURL found: ${CURL_FOUND}")
message(STATUS "CURL version: ${CURL_VERSION}")
message(STATUS "CURL libraries: ${CURL_LIBRARIES}")
message(STATUS "Using json from: ${PROJECT_SOURCE_DIR}/../third_party/json/include")
