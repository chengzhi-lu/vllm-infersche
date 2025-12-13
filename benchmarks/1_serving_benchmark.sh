#!/bin/bash
set -euo pipefail
source $(pwd)/shared_functions.sh
# --------------------------
# 配置参数
# --------------------------
# 基础配置
COUNTER_FILE=".counter.txt"
result_dir="$(pwd)/result"
TOKENIZERS_PARALLELISM="true"

# 模型和数据集配置
model_names=(
  # "meta-llama/Llama-2-13b-chat-hf"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "meta-llama/Llama-2-70b-chat-hf"
  "meta-llama/Llama-3.1-70B-Instruct"
)
parallel_types=(
  # "single"
  "tp"
  # "pp"
)
datasets=(
  # "sharegpt /root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
  # "leval /root/vllm/dataset/paper_assistant_transformed.json"
  "lmsys /root/vllm/dataset/lmsys-chat-1m-aligned.json"
)

# 服务器配置
swap_space=20
preemption_mode="swap"
gpu_memory_utilization=0.45
max_tokens=16384
max_num_seqs=256
max_serving_time=86400
num_shared_blocks=0

request_duration=90
# 测试策略组合
scheduler_swap_policies=(
  "tfittradeoff partial"
#  "fcfs full"
#  "sjf full"
#  "sjmlfq full"
#  "opt full"
)

phases=(
  "hybrid"
  # "decode"
  # "prefill"
)

request_rates=(32)
# request_rates=(4 8)
swap_out_partial_rates=(0.5)

# --------------------------
# 初始化计数器
# --------------------------
if [[ -f "$COUNTER_FILE" ]]; then
  COUNTER=$(cat "$COUNTER_FILE")
else
  COUNTER=0
fi
COUNTER=$((COUNTER + 1))
echo "$COUNTER" >"$COUNTER_FILE"

# --------------------------
# 功能函数定义
# --------------------------

parse_result() {
  local policy=$1
  local swap_policy=$2
  local swap_out_partial_rate=$3
  local request_rate=$4
  local model_name=$5
  local parallel_type=$6

  python3 parse_log.py \
    --policy "$policy" \
    --swap-policy "$swap_policy" \
    --result-dir "$result_dir" \
    --execution-counter "$COUNTER" \
    --request-rate "$request_rate" \
    --swap-out-partial-rate "$swap_out_partial_rate" \
    --model "$model_name" \
    --parallel-type "$parallel_type"
}

# --------------------------
# 主执行流程
# --------------------------
for ptype in "${parallel_types[@]}"; do
  for swap_out_partial_rate in "${swap_out_partial_rates[@]}"; do
    for scheduler_swap_policy in "${scheduler_swap_policies[@]}"; do
      IFS=' ' read -r policy swap_policy <<<"$scheduler_swap_policy"
      for dataset in "${datasets[@]}"; do
        IFS=' ' read -r dataset_name dataset_path <<<"$dataset"
        
        for phase in "${phases[@]}"; do
        for model_name in "${model_names[@]}"; do
          # 跳过70b模型的single并行类型
          if [[ "$model_name" == "meta-llama/Llama-3.1-70B-Instruct" && "$ptype" == "single" ]]; then
            echo "跳过 llama3-70b 的 single 并行类型测试"
            continue
          fi

          if [[ "$model_name" == "meta-llama/Llama-3.1-8B-Instruct" && "$ptype" == "tp" ]]; then
            echo "跳过 llama3-8b 的 tp 并行类型测试"
            continue
          fi

          if [[ "$ptype" == "pp" ]]; then
            host="10.119.46.53"
          else
            host=""
            export RAY_ADDRESS=""
          fi

          # 启动服务
          start_server "$policy" "$swap_policy" "$swap_out_partial_rate" \
            "$ptype" "$model_name" "$phase"

          # 运行基准测试
          for i in {1..1}; do
            for request_rate in "${request_rates[@]}"; do
                max_request_nums=$((request_duration * request_rate))
                  if [[ "$phase" == "prefill" || "$phase" == "decode" ]]; then
                    request_rate=$max_request_nums
                  fi
                  run_benchmark "$policy" "$swap_policy" "$swap_out_partial_rate" \
                    "$request_rate" "$dataset_path" "$dataset_name" \
                    "$model_name" "$ptype" "$max_request_nums" "$phase"
                  sleep 20
                  parse_result "$policy" "$swap_policy" "$swap_out_partial_rate" \
                    "$request_rate" "$model_name" "$ptype"
            done
          done

          # 停止服务器
          echo "停止服务器"
          terminate_server "$ptype"

          sleep 5
          done
        done
      done
    done
  done
done
echo "所有测试完成，结果保存在: $result_dir"
