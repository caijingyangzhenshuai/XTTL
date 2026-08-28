#!/bin/bash
# ============================================================
# 重新编译 third_party/opencv —— 关闭 libtiff 依赖
# ------------------------------------------------------------
# 问题：旧 OpenCV 编译时 WITH_TIFF=ON + BUILD_TIFF=OFF，
#       导致 imgcodecs.so 内部 NEEDED libtiff.so.5，
#       干净容器缺这个库就无法链接/运行。
#
# 方案：WITH_TIFF=OFF 彻底去掉 TIFF 支持（我们的程序只读 jpg/png/bmp，
#       永远不会触发 TIFF 解码），其余编解码器（jpeg/png/webp/openjpeg）
#       全部从源码静态编译进 .so，重编后 OpenCV 完全自包含。
# ============================================================
set -e

OPENCV_DIR="$(cd "$(dirname "$0")" && pwd)/third_party/opencv"
echo "[INFO] OpenCV 源码目录: $OPENCV_DIR"

if [ ! -f "$OPENCV_DIR/CMakeLists.txt" ]; then
    echo "[ERROR] 找不到 OpenCV 源码: $OPENCV_DIR/CMakeLists.txt"
    exit 1
fi

cd "$OPENCV_DIR"

# 备份旧 build（保留以便回退）
if [ -d "build" ]; then
    echo "[INFO] 备份旧 build -> build.old"
    rm -rf build.old
    mv build build.old
fi

echo "[INFO] 开始重新编译 OpenCV (WITH_TIFF=OFF)..."
mkdir -p build && cd build

# 关键策略：
#   - BUILD_LIST: 只编译我们用到的 3 个模块，避免 highgui/videoio/gui 依赖
#   - WITH_TIFF=OFF + BUILD_TIFF=OFF: 彻底去掉 libtiff
#   - BUILD_JPEG/PNG/WEBP/OPENJPEG=ON: 编解码器全部静态编译进 .so
#   - BUILD_ZLIB=ON: zlib 也静态进 .so
#   - 关闭 GUI/video/gstreamer 等一切需要系统库的特性
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" \
    -DBUILD_SHARED_LIBS=ON \
    \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    \
    -DBUILD_TIFF=OFF \
    -DWITH_TIFF=OFF \
    -DBUILD_JPEG=ON \
    -DWITH_JPEG=ON \
    -DBUILD_PNG=ON \
    -DWITH_PNG=ON \
    -DBUILD_WEBP=ON \
    -DWITH_WEBP=ON \
    -DBUILD_OPENJPEG=ON \
    -DWITH_OPENJPEG=ON \
    -DBUILD_ZLIB=ON \
    -DBUILD_OPENEXR=OFF \
    -DWITH_OPENEXR=OFF \
    \
    -DWITH_GTK=OFF \
    -DWITH_GTK_2_X=OFF \
    -DWITH_VTK=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DWITH_PROTOBUF=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_AVIF=OFF \
    -DWITH_IMGCODEC_HDR=OFF \
    -DWITH_IMGCODEC_PFM=OFF \
    -DWITH_IMGCODEC_PXM=OFF \
    -DWITH_IMGCODEC_SUNRASTER=OFF \
    -DWITH_IMGCODEC_GIF=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo ""
echo "[INFO] 执行 make (可能需要几分钟)..."
make -j"$(nproc)"

echo ""
echo "[INFO] 执行 make install..."
make install

# 确保 lib/lib64 两个目录都有（兼容旧探测逻辑）
cd "$OPENCV_DIR"
if [ -d "lib" ] && [ ! -d "lib64" ]; then
    ln -sf lib lib64
    echo "[INFO] 创建 lib64 -> lib 符号链接（兼容探测逻辑）"
fi

echo ""
echo "=========================================="
echo "  OpenCV 重编译完成"
echo "=========================================="
echo "产物位置: $OPENCV_DIR/lib"
echo ""
echo "验证 libtiff 依赖已去除:"
LIB64="$OPENCV_DIR/lib64"
[ -L "$LIB64" ] && LIB64="$OPENCV_DIR/lib"
if command -v readelf >/dev/null 2>&1; then
    echo "--- imgcodecs 的 NEEDED 依赖 ---"
    readelf -d "$LIB64/libopencv_imgcodecs.so" 2>/dev/null | grep NEEDED || echo "(无法读取)"
    echo ""
    echo "--- 是否还有 libtiff ---"
    if readelf -d "$LIB64/libopencv_imgcodecs.so" 2>/dev/null | grep -qi tiff; then
        echo "[警告] 仍依赖 libtiff，请检查 cmake 配置"
    else
        echo "[OK] imgcodecs 已不再依赖 libtiff"
    fi
fi
echo ""
echo "下一步: 重新编译你的项目"
echo "  cd $(cd "$OPENCV_DIR/../.." && pwd)"
echo "  bash install_cann.sh"
