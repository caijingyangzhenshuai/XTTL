#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

log_info()  { echo -e "\033[32m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

check_root() {
    log_info "检查用户权限..."
    if [ "$(id -u)" != "0" ]; then
        log_error "请使用root用户执行此脚本"
        exit 1
    fi
}

check_arch() {
    log_info "检查系统架构..."
    local arch=$(uname -m)
    if [ "$arch" != "aarch64" ]; then
        log_error "此脚本仅支持aarch64架构，当前: $arch"
        exit 1
    fi
    log_info "架构: $arch"
}

check_npu_driver() {
    log_info "检查NPU驱动..."
    
    if [ ! -d "/usr/local/Ascend/driver" ]; then
        log_error "Ascend Driver 未安装"
        exit 1
    fi
    
    if [ ! -c "/dev/davinci0" ]; then
        log_error "NPU设备不存在"
        exit 1
    fi
    
    if command -v npu-smi &> /dev/null; then
        log_info "NPU状态:"
        npu-smi info
    fi
}

check_cann() {
    log_info "检查CANN Toolkit..."
    
    local toolkit_home=""
    
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    fi
    
    for path in "$ASCEND_TOOLKIT_HOME" "/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/cann-8.5.1"; do
        if [ -d "$path" ] && [ -f "$path/aarch64-linux/include/acl/acl.h" ]; then
            toolkit_home="$path"
            break
        fi
    done
    
    if [ -z "$toolkit_home" ]; then
        log_error "CANN Toolkit 未安装"
        exit 1
    fi
    
    log_info "CANN路径: $toolkit_home"
    
    if [ -f "$toolkit_home/version.info" ]; then
        local version=$(grep -E "^Version" "$toolkit_home/version.info" 2>/dev/null | cut -d'=' -f2)
        [ -n "$version" ] && log_info "CANN版本: $version"
    fi
    
    export ASCEND_TOOLKIT_HOME="$toolkit_home"
}

check_build_tools() {
    log_info "检查编译工具..."
    
    local missing=()
    
    if ! command -v cmake &> /dev/null; then
        missing+=("cmake")
    else
        log_info "CMake: $(cmake --version | head -1 | grep -oP 'version \K[0-9.]+')"
    fi
    
    if ! command -v gcc &> /dev/null; then
        missing+=("gcc")
    else
        log_info "GCC: $(gcc --version | head -1 | grep -oP '\K[0-9]+\.[0-9]+\.[0-9]+')"
    fi
    
    if ! command -v g++ &> /dev/null; then
        missing+=("g++")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "缺少编译工具: ${missing[*]}"
        log_info "安装命令: yum install -y ${missing[*]}"
        exit 1
    fi
}

check_opencv() {
    log_info "检查OpenCV..."
    
    # 方法1: pkg-config 检测
    if pkg-config --exists opencv4 2>/dev/null; then
        log_info "OpenCV: $(pkg-config --modversion opencv4)"
        return 0
    elif pkg-config --exists opencv 2>/dev/null; then
        log_info "OpenCV: $(pkg-config --modversion opencv)"
        return 0
    fi
    
    # 方法2: 在多个标准路径查找库文件
    local search_paths=(
        "/usr/lib"
        "/usr/lib64"
        "/usr/local/lib"
        "/usr/local/lib64"
        "/opt/opencv/lib"
        "/root/anaconda3/lib"
        "/root/miniconda3/lib"
    )
    
    local opencv_lib=""
    for path in "${search_paths[@]}"; do
        opencv_lib=$(find "$path" -name "libopencv_core.so*" 2>/dev/null | head -1)
        if [ -n "$opencv_lib" ]; then
            log_info "OpenCV库: $opencv_lib"
            return 0
        fi
    done
    
    # 方法3: 检查项目自带的第三方OpenCV库
    local third_party_opencv=(
        "$PROJECT_ROOT/third_party/opencv/build"
        "$PROJECT_ROOT/../third_party/opencv/build"
        "/home/HwHiAiUser/cann/third_party/opencv/build"
        "/home/HwHiAiUser/third_party/opencv/build"
    )
    
    for opencv_path in "${third_party_opencv[@]}"; do
        if [ -d "$opencv_path" ]; then
            # 检查是否有 lib 目录和 include 目录
            if [ -d "$opencv_path/lib" ] && [ -d "$opencv_path/include" ]; then
                # 检查是否有核心库文件
                local lib_file=$(find "$opencv_path/lib" -name "libopencv_core.so*" 2>/dev/null | head -1)
                if [ -n "$lib_file" ]; then
                    log_info "OpenCV (第三方库): $opencv_path"
                    log_info "OpenCV库文件: $lib_file"
                    return 0
                fi
            fi
        fi
    done
    
    # 方法4: 检查头文件
    local opencv_include=$(find /usr /usr/local /opt -name "opencv2/opencv.hpp" 2>/dev/null | head -1)
    if [ -n "$opencv_include" ]; then
        log_info "OpenCV头文件: $opencv_include"
        return 0
    fi
    
    # 方法5: 检查运行时链接
    if command -v ldd &> /dev/null; then
        local opencv_runtime=$(ldconfig -p 2>/dev/null | grep "libopencv_core" | head -1)
        if [ -n "$opencv_runtime" ]; then
            log_info "OpenCV运行时库: $(echo $opencv_runtime | awk '{print $NF}')"
            return 0
        fi
    fi
    
    log_warn "OpenCV未安装或未正确配置，某些功能可能受限"
    log_info "建议安装: yum install -y opencv-devel"
    return 1
}

build_project() {
    log_info "编译项目..."
    
    cd "$PROJECT_ROOT/src"
    
    rm -rf build
    mkdir -p build && cd build
    
    cmake .. || { log_error "cmake失败"; exit 1; }
    make -j$(nproc) || { log_error "make失败"; exit 1; }
    
    log_info "编译完成"
}

install_executable() {
    log_info "安装可执行文件..."
    
    local exe_name="kzzk_cv"
    local exe_path="$PROJECT_ROOT/src/build/$exe_name"
    
    if [ ! -f "$exe_path" ]; then
        log_error "未找到可执行文件: $exe_path"
        exit 1
    fi
    
    cp "$exe_path" "/usr/local/bin/$exe_name"
    chmod +x "/usr/local/bin/$exe_name"
    
    log_info "已安装: /usr/local/bin/$exe_name"
}

show_result() {
    echo ""
    echo "=========================================="
    echo "       CANN 环境部署完成"
    echo "=========================================="
    echo ""
    echo "检查项:"
    echo "  ✓ 用户权限: root"
    echo "  ✓ 系统架构: aarch64"
    echo "  ✓ NPU驱动"
    echo "  ✓ CANN Toolkit"
    echo "  ✓ 编译工具"
    echo ""
    echo "使用命令:"
    echo "  kzzk_cv --help"
    echo "  kzzk_cv -m /path/to/model.om -i /path/to/image.jpg"
    echo "=========================================="
}

main() {
    echo "=========================================="
    echo "       CANN 环境部署脚本"
    echo "=========================================="
    echo ""
    
    check_root
    check_arch
    check_npu_driver
    check_cann
    check_build_tools
    check_opencv
    
    echo ""
    build_project
    install_executable
    
    show_result
}

main "$@"
