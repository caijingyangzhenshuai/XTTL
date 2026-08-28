#!/bin/bash
set -e

# ============================================================
# MindIE 模型执行脚本
# 功能：选择模型 → 编译客户端 → 修改配置文件 → 启动服务
# ============================================================

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/home/HwHiAiUser/mindie/cpp/build"
EXE_NAME="kzzk_llm"
CONFIG_FILE="/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json"
DAEMON_BIN="/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon"

log_info()  { echo -e "\033[32m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

check_root() {
    if [ "$(whoami)" != "root" ]; then
        log_error "请以root用户运行此脚本"
        exit 1
    fi
}

# ============================================================
# 选择模型
# ============================================================
select_model() {
    local model_dir="/home/HwHiAiUser/mindie/model"
    
    if [ ! -d "$model_dir" ]; then
        log_error "模型目录不存在: $model_dir"
        exit 1
    fi
    
    local models=()
    echo ""
    echo "可用模型列表:"
    echo "----------------------------------------"
    
    for dir in "$model_dir"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            if [ "$name" != "kernel_meta" ]; then
                models+=("$name")
                echo "  ${#models[@]}. $name"
            fi
        fi
    done
    
    echo "----------------------------------------"
    
    if [ ${#models[@]} -eq 0 ]; then
        log_error "未找到模型"
        exit 1
    fi
    
    echo ""
    read -p "请输入选择 [默认1]: " choice
    
    local idx=${choice:-1}
    idx=$((idx - 1))
    
    if [ $idx -lt 0 ] || [ $idx -ge ${#models[@]} ]; then
        log_warn "无效选择，使用默认模型: ${models[0]}"
        idx=0
    fi
    
    MODEL_NAME="${models[$idx]}"
    MODEL_PATH="$model_dir/$MODEL_NAME"
    
    log_info "已选择模型: $MODEL_NAME"
    log_info "模型路径: $MODEL_PATH"
}

# ============================================================
# 获取 NPU 设备 ID
# ============================================================
get_npu_ids() {
    # 动态检测 NPU 设备数量
    local total_npu=$(ls -1 /dev/davinci* 2>/dev/null | grep -c '/dev/davinci[0-9]' || echo "0")
    
    if [ "$total_npu" -eq 0 ]; then
        log_error "未检测到NPU设备"
        exit 1
    fi
    
    echo ""
    echo "当前可用 NPU 设备: 0 ~ $((total_npu - 1))"
    read -p "请输入要使用的 NPU 设备ID（多个用逗号分隔，如 0,1，直接回车默认 0,1）: " user_input
    
    if [ -z "$user_input" ]; then
        user_input="0,1"
        log_info "使用默认 NPU 设备: $user_input"
    fi
    
    # 验证输入
    local valid=true
    IFS=',' read -ra ids <<< "$user_input"
    for id in "${ids[@]}"; do
        id=$(echo "$id" | xargs)
        if ! [[ "$id" =~ ^[0-9]+$ ]] || [ "$id" -ge "$total_npu" ]; then
            valid=false
            log_error "无效的 NPU 设备ID: $id"
        fi
    done
    
    if ! $valid; then
        exit 1
    fi
    
    NPU_DEVICE_IDS="$user_input"
    NPU_COUNT=$(echo "$NPU_DEVICE_IDS" | tr ',' '\n' | wc -l)
    
    log_info "NPU设备ID: [$NPU_DEVICE_IDS]"
    log_info "worldSize: $NPU_COUNT"
}

# ============================================================
# 修改配置文件
# ============================================================
update_config() {
    log_info "更新配置文件..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
    log_info "已备份: ${CONFIG_FILE}.bak"
    
    python3 << PYEOF
import json, re

with open("$CONFIG_FILE", "r") as f:
    cfg = json.load(f)

# 1. httpsEnabled 改为 false
cfg["ServerConfig"]["httpsEnabled"] = False

# 2. npuDeviceIds 改为用户输入的 ID
cfg["BackendConfig"]["npuDeviceIds"] = [[$NPU_DEVICE_IDS]]

# 3. modelName
cfg["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelName"] = "$MODEL_NAME"

# 4. modelWeightPath
cfg["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"] = "$MODEL_PATH"

# 5. worldSize
cfg["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["worldSize"] = $NPU_COUNT

text = json.dumps(cfg, indent=4)
text = re.sub(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', r'[\1,\2]', text)

with open("$CONFIG_FILE", "w") as f:
    f.write(text)
    f.write("\n")
PYEOF

    log_info "配置文件更新完成"
}

# ============================================================
# 编译客户端
# ============================================================
build_client() {
    log_info "编译客户端..."
    mkdir -p "$BUILD_DIR"
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"/*
    fi
    cd "$BUILD_DIR"
    cmake .. && make -j$(nproc)
    
    local exe_src="$BUILD_DIR/bin/$EXE_NAME"
    if [ -f "$exe_src" ]; then
        cp "$exe_src" /usr/local/bin/
        chmod +x /usr/local/bin/$EXE_NAME
        log_info "客户端已部署: /usr/local/bin/$EXE_NAME"
    else
        exe_src="$BUILD_DIR/$EXE_NAME"
        if [ -f "$exe_src" ]; then
            cp "$exe_src" /usr/local/bin/
            chmod +x /usr/local/bin/$EXE_NAME
            log_info "客户端已部署: /usr/local/bin/$EXE_NAME"
        else
            log_warn "未找到可执行文件 $EXE_NAME，跳过部署"
            log_warn "请检查编译输出目录: $BUILD_DIR"
        fi
    fi
    
    cd "$WORKDIR"
}

# ============================================================
# 启动服务
# ============================================================
start_service() {
    log_info "启动服务..."
    
    if pgrep -f mindieservice_daemon > /dev/null 2>&1; then
        log_info "检测到已有服务在运行，先停止旧服务..."
        SCRIPT_PID=$$
        
        # 1. 杀掉 mindieservice 主进程
        pkill -9 -f mindieservice_daemon 2>/dev/null || true
        pkill -9 -f mindie-service 2>/dev/null || true
        sleep 2
        
        # 2. 杀掉所有 Python 进程（包括推理子进程 mindie_llm_backend）
        pkill -9 -f python 2>/dev/null || true
        sleep 2
        
        # 3. 强制清理 NPU 残留进程
        ps aux | grep -E 'mindie|tbe|atb|ge' | grep -v grep | grep -v "run_model" | grep -v "bash" | awk '{print $2}' | while read pid; do
            if [ -n "$pid" ] && [ "$pid" != "$SCRIPT_PID" ]; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        
        sleep 2
        
        # 4. 清理共享内存
        rm -rf /dev/shm/mindie_* 2>/dev/null || true
        rm -rf /tmp/mindie_* 2>/dev/null || true
        
        log_info "旧服务已停止"
    fi
    
    mkdir -p /var/log/mindie
    
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "模型目录不存在: $MODEL_PATH"
        exit 1
    fi
    
    local mindie_home="/usr/local/Ascend/mindie/latest"
    cd "$mindie_home"
    
    source /usr/local/Ascend/mindie/set_env.sh
    log_info "环境变量已设置"
    
    local log_file="/var/log/mindie/service.log"
    > "$log_file"
    
    nohup $DAEMON_BIN > "$log_file" 2>&1 &
    local daemon_pid=$!
    log_info "服务进程 PID: $daemon_pid"
    
    local elapsed=0
    local max_wait=60
    local bar_width=40
    
    while [ $elapsed -lt $max_wait ]; do
        sleep 2
        elapsed=$((elapsed + 2))
        
        if ! kill -0 $daemon_pid 2>/dev/null; then
            echo ""
            log_error "服务启动失败"
            echo ""
            echo "=============== 完整错误日志 ==============="
            cat "$log_file"
            echo "============================================="
            cd "$WORKDIR"
            exit 1
        fi
        
        if grep -q "Daemon start success" "$log_file" 2>/dev/null; then
            printf "\r[%-${bar_width}s] %3d%% 服务启动成功\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
            log_info "服务启动成功"
            break
        fi
        
        local last_log=$(tail -3 "$log_file" 2>/dev/null | grep -v '^$' | tail -1)
        local progress=$((elapsed * bar_width / max_wait))
        local percentage=$((elapsed * 100 / max_wait))
        local filled=$(printf '=%.0s' $(seq 1 $progress))
        local empty=$(printf ' %.0s' $(seq 1 $((bar_width - progress))))
        printf "\r[%s%s] %3d%% %s" "$filled" "$empty" $percentage "${last_log:0:60}"
    done
    
    cd "$WORKDIR"
}

# ============================================================
# 等待服务就绪
# ============================================================
wait_service() {
    log_info "等待服务就绪..."
    
    local daemon_pid=$(pgrep -f mindieservice_daemon 2>/dev/null | head -1)
    if [ -z "$daemon_pid" ]; then
        log_error "服务进程不存在"
        exit 1
    fi
    
    local elapsed=0
    local max_wait=300
    local bar_width=40
    local log_file="/var/log/mindie/service.log"
    
    while [ $elapsed -lt $max_wait ]; do
        if ! kill -0 $daemon_pid 2>/dev/null; then
            echo ""
            log_error "服务进程已崩溃"
            exit 1
        fi
        
        # 检查 curl 或日志中的 Daemon start success
        if curl -s http://127.0.0.1:1025/v1/models > /dev/null 2>&1; then
            printf "\r[%-${bar_width}s] %3d%% 服务就绪\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
            log_info "服务就绪"
            return 0
        fi
        
        if grep -q "Daemon start success" "$log_file" 2>/dev/null; then
            printf "\r[%-${bar_width}s] %3d%% 服务就绪\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
            log_info "服务就绪"
            return 0
        fi
        
        # 获取最新日志
        local last_log=$(tail -5 "$log_file" 2>/dev/null | grep -v '^$' | tail -1)
        
        local progress=$((elapsed * bar_width / max_wait))
        local percentage=$((elapsed * 100 / max_wait))
        local filled=$(printf '=%.0s' $(seq 1 $progress))
        local empty=$(printf ' %.0s' $(seq 1 $((bar_width - progress))))
        
        printf "\r[%s%s] %3d%% %s" "$filled" "$empty" $percentage "${last_log:0:60}"
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    echo ""
    log_error "服务未就绪，已等待 $max_wait 秒"
    tail -20 "$log_file"
    exit 1
}

# ============================================================
# 显示结果
# ============================================================
show_result() {
    echo ""
    echo "=========================================="
    echo "       MindIE 模型部署完成"
    echo "=========================================="
    echo ""
    echo "部署配置:"
    echo "  - 模型名称: $MODEL_NAME"
    echo "  - 模型路径: $MODEL_PATH"
    echo "  - NPU设备: [$NPU_DEVICE_IDS]"
    echo "  - worldSize: $NPU_COUNT"
    echo "  - 配置文件: $CONFIG_FILE"
    echo "  - 日志文件: /var/log/mindie/service.log"
    echo ""
    echo "使用命令:"
    echo "  curl http://127.0.0.1:1025/v1/models"
    echo "=========================================="
}

# ============================================================
# 主流程
# ============================================================
main() {
    echo "=========================================="
    echo "       MindIE 模型执行脚本"
    echo "=========================================="
    echo ""
    
    check_root
    select_model
    get_npu_ids
    update_config
    build_client
    start_service
    wait_service && show_result
}

main "$@"