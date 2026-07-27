#!/bin/bash
set -e

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/home/HwHiAiUser/mindie/src/build"
EXE_NAME="kzzk_llm"
CONFIG_FILE="/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json"
DAEMON_BIN="/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon"

NPU_COUNT=0
NPU_DEVICE_IDS=""

log_info()  { echo -e "\033[32m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

check_root() {
    if [ "$(whoami)" != "root" ]; then
        log_error "请以root用户运行此脚本"
        exit 1
    fi
}

detect_npu() {
    log_info "检测NPU设备..."
    
    if [ ! -c "/dev/davinci0" ]; then
        log_error "NPU设备不存在"
        exit 1
    fi
    
    NPU_COUNT=$(ls -1 /dev/davinci* 2>/dev/null | grep -c '/dev/davinci[0-9]' || echo "0")
    
    if [ "$NPU_COUNT" -eq 0 ]; then
        log_error "未检测到NPU设备"
        exit 1
    fi
    
    log_info "检测到 $NPU_COUNT 个 NPU 设备"
    
    NPU_DEVICE_IDS=""
    for ((i=0; i<NPU_COUNT; i++)); do
        if [ -n "$NPU_DEVICE_IDS" ]; then
            NPU_DEVICE_IDS="$NPU_DEVICE_IDS,$i"
        else
            NPU_DEVICE_IDS="$i"
        fi
    done
    
    log_info "NPU设备ID: [$NPU_DEVICE_IDS]"
    
    if command -v npu-smi &> /dev/null; then
        log_info "NPU状态:"
        npu-smi info
    fi
}

check_mindie() {
    local mindie_home="/usr/local/Ascend/mindie/latest"
    if [ ! -d "$mindie_home" ]; then
        log_error "MindIE未安装: $mindie_home"
        exit 1
    fi
    if [ ! -f "$DAEMON_BIN" ]; then
        log_error "服务启动脚本不存在: $DAEMON_BIN"
        exit 1
    fi
    log_info "MindIE路径: $mindie_home"
}

list_models() {
    log_info "扫描可用模型..."
    
    local model_dir="/home/HwHiAiUser/mindie/model"
    
    if [ ! -d "$model_dir" ]; then
        log_error "模型目录不存在: $model_dir"
        exit 1
    fi
    
    local models=()
    local idx=1
    
    echo ""
    echo "可用模型列表:"
    echo "----------------------------------------"
    
    for dir in "$model_dir"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            # 排除 kernel_meta 目录（不是模型）
            if [ "$name" != "kernel_meta" ]; then
                models+=("$name")
                echo "  $idx. $name"
                ((idx++))
            fi
        fi
    done
    
    echo "----------------------------------------"
    
    if [ ${#models[@]} -eq 0 ]; then
        log_error "未找到模型，请检查 $model_dir 目录"
        exit 1
    fi
    
    MODEL_LIST=("${models[@]}")
}

select_model() {
    list_models
    
    # 从 config.json 中读取当前配置的模型名称
    local CURRENT_MODEL=""
    if [ -f "$CONFIG_FILE" ]; then
        CURRENT_MODEL=$(grep '"modelName"' "$CONFIG_FILE" | sed 's/.*: "\(.*\)".*/\1/')
    fi
    
    # 显示当前配置的模型
    if [ -n "$CURRENT_MODEL" ]; then
        echo ""
        echo "当前配置文件中的模型为：$CURRENT_MODEL"
    fi
    
    echo ""
    read -p "请输入选择 [默认1]: " choice
    
    local idx=${choice:-1}
    idx=$((idx - 1))
    
    if [ $idx -lt 0 ] || [ $idx -ge ${#MODEL_LIST[@]} ]; then
        log_warn "无效选择，使用默认模型: ${MODEL_LIST[0]}"
        idx=0
    fi
    
    MODEL_NAME="${MODEL_LIST[$idx]}"
    MODEL_PATH="/home/HwHiAiUser/mindie/model/$MODEL_NAME"
    
    log_info "已选择模型: $MODEL_NAME"
    log_info "模型路径: $MODEL_PATH"
    
    # 如果选择的模型与当前配置不同，则更新 config.json
    if [ -n "$CURRENT_MODEL" ] && [ "$CURRENT_MODEL" != "$MODEL_NAME" ]; then
        log_info "检测到模型变更，更新配置文件..."
        
        # 修改 config.json 中的 modelName 和 modelWeightPath
        if [ -f "$CONFIG_FILE" ]; then
            # 备份 config.json
            cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
            
            # 使用 sed 修改 modelName
            sed -i 's/"modelName"[[:space:]]*:[[:space:]]*"[^"]*"/"modelName": "'"$MODEL_NAME"'"/g' "$CONFIG_FILE"
            
            # 使用 sed 修改 modelWeightPath
            sed -i 's|"modelWeightPath"[[:space:]]*:[[:space:]]*"[^"]*"|"modelWeightPath": "'"$MODEL_PATH"'"|g' "$CONFIG_FILE"
            
            log_info "已更新 config.json：modelName=$MODEL_NAME, modelWeightPath=$MODEL_PATH"
            log_info "备份文件: ${CONFIG_FILE}.backup"
        fi
    fi
    
    # 验证模型目录是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "模型目录不存在: $MODEL_PATH"
        exit 1
    fi
    
    log_info "使用模型: $MODEL_NAME"
    log_info "模型路径: $MODEL_PATH"
}

build_project() {
    log_info "编译工程..."
    mkdir -p "$BUILD_DIR"
    rm -rf "$BUILD_DIR"/*
    cd "$BUILD_DIR"
    cmake .. && make -j$(nproc)
    
    if [ -f "$BUILD_DIR/$EXE_NAME" ]; then
        cp "$BUILD_DIR/$EXE_NAME" /usr/local/bin/
        chmod +x /usr/local/bin/$EXE_NAME
        log_info "可执行文件已部署: /usr/local/bin/$EXE_NAME"
    fi
    cd /
}

generate_config() {
    log_info "更新配置文件..."
    
    # 检查配置文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    # 备份原配置文件
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
    log_info "已备份原配置文件: ${CONFIG_FILE}.bak"
    
    # 使用 sed 修改配置文件中的关键字段（仿照 install_all_offline.sh 的方式）
    
    # 修改 modelName
    sed -i 's/"modelName"[[:space:]]*:[[:space:]]*"[^"]*"/"modelName": "'"$MODEL_NAME"'"/g' "$CONFIG_FILE"
    log_info "已修改 modelName: $MODEL_NAME"
    
    # 修改 modelWeightPath
    sed -i 's|"modelWeightPath"[[:space:]]*:[[:space:]]*"[^"]*"|"modelWeightPath": "'"$MODEL_PATH"'"|g' "$CONFIG_FILE"
    log_info "已修改 modelWeightPath: $MODEL_PATH"
    
    # 修改 npuDeviceIds
    sed -i 's/"npuDeviceIds"[[:space:]]*:[[:space:]]*\[\[.*\]\]/"npuDeviceIds": [['"$NPU_DEVICE_IDS"']]/g' "$CONFIG_FILE"
    log_info "已修改 npuDeviceIds: [[$NPU_DEVICE_IDS]]"
    
    # 修改 worldSize
    sed -i 's/"worldSize"[[:space:]]*:[[:space:]]*[0-9]*/"worldSize": '"$NPU_COUNT"'/g' "$CONFIG_FILE"
    log_info "已修改 worldSize: $NPU_COUNT"
    
    # 修改 modelInstanceNumber
    sed -i 's/"modelInstanceNumber"[[:space:]]*:[[:space:]]*[0-9]*/"modelInstanceNumber": 1/g' "$CONFIG_FILE"
    log_info "已修改 modelInstanceNumber: 1"
    
    # 禁用 TLS 相关配置（避免证书文件不存在导致启动失败）
    # 设置 httpsEnabled 为 false
    sed -i 's/"httpsEnabled"[[:space:]]*:[[:space:]]*true/"httpsEnabled": false/g' "$CONFIG_FILE"
    log_info "已禁用 HTTPS: httpsEnabled=false"
    
    # 设置 interCommTLSEnabled 为 false
    sed -i 's/"interCommTLSEnabled"[[:space:]]*:[[:space:]]*true/"interCommTLSEnabled": false/g' "$CONFIG_FILE"
    log_info "已禁用内部通信 TLS: interCommTLSEnabled=false"
    
    # 设置 interNodeTLSEnabled 为 false
    sed -i 's/"interNodeTLSEnabled"[[:space:]]*:[[:space:]]*true/"interNodeTLSEnabled": false/g' "$CONFIG_FILE"
    log_info "已禁用节点间 TLS: interNodeTLSEnabled=false"
    
    # 设置推理超时时间（解决 e2eTimeout 问题）
    sed -i 's/"e2eTimeout"[[:space:]]*:[[:space:]]*[0-9]*/"e2eTimeout": 600/g' "$CONFIG_FILE"
    log_info "已设置推理超时时间: e2eTimeout=600秒"
    
    # 设置引擎回调超时时间
    sed -i 's/"engineCallbackTimeout"[[:space:]]*:[[:space:]]*[0-9]*/"engineCallbackTimeout": 600/g' "$CONFIG_FILE"
    log_info "已设置引擎回调超时时间: engineCallbackTimeout=600秒"

    # 设置 token 生成超时时间（解决 tokenTimeout 问题）
    if grep -q '"tokenTimeout"' "$CONFIG_FILE"; then
        sed -i 's/"tokenTimeout"[[:space:]]*:[[:space:]]*[0-9]*/"tokenTimeout": 600/g' "$CONFIG_FILE"
        log_info "已设置 token 生成超时时间: tokenTimeout=600秒"
    else
        # 如果配置文件中没有 tokenTimeout 字段，在 engineCallbackTimeout 后添加
        sed -i '/"engineCallbackTimeout"/a\    "tokenTimeout": 600,' "$CONFIG_FILE"
        log_info "已添加 token 生成超时时间: tokenTimeout=600秒"
    fi
    
    # 验证修改结果
    log_info "配置文件修改完成，验证修改结果:"
    echo "  modelName: $(grep '"modelName"' "$CONFIG_FILE" | sed 's/.*: "\(.*\)".*/\1/')"
    echo "  modelWeightPath: $(grep '"modelWeightPath"' "$CONFIG_FILE" | sed 's/.*: "\(.*\)".*/\1/')"
    echo "  npuDeviceIds: $(grep '"npuDeviceIds"' "$CONFIG_FILE" | sed 's/.*: \(\[.*\]\)/\1/')"
    echo "  worldSize: $(grep '"worldSize"' "$CONFIG_FILE" | sed 's/.*: \([0-9]*\)/\1/')"
    echo "  httpsEnabled: $(grep '"httpsEnabled"' "$CONFIG_FILE" | sed 's/.*: \(true\|false\)/\1/')"
}

stop_service() {
    log_info "停止已有服务..."
    
    SCRIPT_PID=$$
    log_info "当前脚本 PID: $SCRIPT_PID"
    
    log_info "停止 mindieservice 进程..."
    pkill -9 -f mindieservice_daemon 2>/dev/null || true
    pkill -9 -f mindie-service 2>/dev/null || true
    sleep 2
    
    log_info "停止所有相关 Python/NPU 进程..."
    pkill -9 -f python 2>/dev/null || true
    
    sleep 2
    
    log_info "强制清理 NPU 残留进程..."
    ps aux | grep -E 'mindie|tbe|atb|ge' | grep -v grep | grep -v "install_mindie" | grep -v "bash" | awk '{print $2}' | while read pid; do
        if [ -n "$pid" ] && [ "$pid" != "$SCRIPT_PID" ]; then
            log_info "杀掉进程: $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 2
    
    # 清理共享内存
    log_info "清理共享内存..."
    rm -rf /dev/shm/mindie_* 2>/dev/null || true
    rm -rf /tmp/mindie_* 2>/dev/null || true
    
    log_info "服务已停止"
}

setup_env() {
    log_info "设置环境变量..."
    
    local cann_path=$(ls -d /usr/local/Ascend/ascend-toolkit/* 2>/dev/null | head -1)
    [ -f "$cann_path/set_env.sh" ] && source "$cann_path/set_env.sh"
    [ -f "/usr/local/Ascend/mindie/set_env.sh" ] && source "/usr/local/Ascend/mindie/set_env.sh"
    
    export ASCEND_HOME=/usr/local/Ascend
    export ASCEND_DEVICE_ID=0
    export ASCEND_DEVICE_NUM=$NPU_COUNT
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export RANK_SIZE=$NPU_COUNT
    export RANK_ID=0
    export LOCAL_RANK=0
    export HCCL_WHITELIST_DISABLE=1
    export HCCL_CONNECT_TIMEOUT=120
    
    # 设置日志级别
    export MINDIE_LOG_LEVEL=INFO
    export MINDIE_LOG_TO_STDOUT=1
    
    export LD_LIBRARY_PATH="/usr/local/Ascend/mindie/latest/lib:/usr/local/Ascend/mindie/latest/mindie-service/lib:$LD_LIBRARY_PATH"
    
    if [ -d "$cann_path" ]; then
        export LD_LIBRARY_PATH="$cann_path/lib64:$cann_path/opskernel/lib64:$cann_path/compiler/lib64:$LD_LIBRARY_PATH"
    fi
    
    # 查找并添加 libtorch 路径
    local torch_path=$(find /usr/local/Ascend/mindie -name "libtorch.so" -type f 2>/dev/null | head -1)
    if [ -n "$torch_path" ]; then
        export LD_LIBRARY_PATH="$(dirname $torch_path):$LD_LIBRARY_PATH"
    fi
    
    log_info "环境变量设置完成"
    log_info "ASCEND_DEVICE_NUM=$ASCEND_DEVICE_NUM"
    log_info "RANK_SIZE=$RANK_SIZE"
}

start_service() {
    log_info "启动服务..."
    mkdir -p /var/log/mindie
    
    # 检查模型文件
    log_info "检查模型文件..."
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "模型目录不存在: $MODEL_PATH"
        exit 1
    fi
    
    local model_files=$(find "$MODEL_PATH" -type f 2>/dev/null | wc -l)
    log_info "模型目录包含 $model_files 个文件"
    
    # 清理可能残留的文件
    rm -f /tmp/mindie_* 2>/dev/null || true
    
    nohup $DAEMON_BIN > /var/log/mindie/service.log 2>&1 &
    sleep 15
    
    if pgrep -f mindieservice > /dev/null; then
        log_info "服务已启动 (PID: $(pgrep -f mindieservice))"
    else
        log_error "服务启动失败"
        echo ""
        echo "=============== 错误日志 ==============="
        tail -50 /var/log/mindie/service.log
        echo "=========================================="
        echo ""
        echo "可能的原因:"
        echo "  1. 模型文件损坏或不完整"
        echo "  2. NPU 内存不足"
        echo "  3. 模型与 MindIE 版本不兼容"
        echo "  4. 配置文件格式错误"
        echo ""
        echo "建议检查:"
        echo "  - 模型文件: ls -la $MODEL_PATH"
        echo "  - NPU 状态: npu-smi info"
        echo "  - 完整日志: cat /var/log/mindie/service.log"
        exit 1
    fi
}

wait_service() {
    log_info "等待服务就绪..."
    
    local elapsed=0
    local max_wait=300
    local bar_width=40
    
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://127.0.0.1:1025/v1/models > /dev/null 2>&1; then
            # 服务就绪，显示完整进度条
            printf "\r[%-${bar_width}s] %3d%% 服务就绪\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
            log_info "服务就绪"
            return 0
        fi
        
        if grep -q "Daemon start success" /var/log/mindie/service.log 2>/dev/null; then
            # 服务就绪，显示完整进度条
            printf "\r[%-${bar_width}s] %3d%% 服务就绪\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
            log_info "服务就绪"
            return 0
        fi
        
        # 计算进度条
        local progress=$((elapsed * bar_width / max_wait))
        local percentage=$((elapsed * 100 / max_wait))
        local filled=$(printf '=%.0s' $(seq 1 $progress))
        local empty=$(printf ' %.0s' $(seq 1 $((bar_width - progress))))
        
        printf "\r[%s%s] %3d%% 等待中..." "$filled" "$empty" $percentage
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    # 超时，显示完整进度条
    printf "\r[%-${bar_width}s] %3d%% 超时\n" "$(printf '=%.0s' $(seq 1 $bar_width))" 100
    log_error "服务未就绪，已等待 $max_wait 秒"
    tail -20 /var/log/mindie/service.log
    return 1
}

show_result() {
    echo ""
    echo "=========================================="
    echo "       MindIE 模型部署完成"
    echo "=========================================="
    echo ""
    echo "部署配置:"
    echo "  - NPU数量: $NPU_COUNT"
    echo "  - NPU设备: [$NPU_DEVICE_IDS]"
    echo "  - 模型名称: $MODEL_NAME"
    echo "  - 模型路径: $MODEL_PATH"
    echo ""
    echo "文件位置:"
    echo "  - 配置文件: $CONFIG_FILE"
    echo "  - 日志文件: /var/log/mindie/service.log"
    echo ""
    echo "使用命令:"
    echo "  kzzk_llm --listModels"
    echo "  kzzk_llm --modelfile $MODEL_NAME --prompt \"你好\""
    echo "=========================================="
}

main() {
    echo "=========================================="
    echo "       MindIE 容器内部署脚本"
    echo "=========================================="
    echo ""
    
    check_root
    detect_npu
    check_mindie
    select_model
    build_project
    generate_config
    stop_service
    setup_env
    start_service
    wait_service && show_result
}

main "$@"
