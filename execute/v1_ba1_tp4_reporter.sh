SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

target_model_path="/mnt/geminisgceph1/geminicephfs/mmsearch-ai-research/models/Llama-3.1-8B-Instruct"
#draft_model_path=${ROOT_DIR}/outputs/QwQ-32B-eagle3/epoch_9
# draft_model_path=${ROOT_DIR}/outputs/QwQ-32B-eagle3_v1/epoch_0
# draft_model_path=${ROOT_DIR}/outputs2/QwQ-32B-eagle3_v2/epoch_0
draft_model_path="/mnt/geminisgceph1/geminicephfs/mmsearch-ai-research/users/juqiuwang/sglang-EAGLE3-Llama-3.1-Instruct-8B"
#config_list: batch_size, num_steps, topk, num_verify_tokens     "1,0,0,0"
#1,0,0,0    1,1,1,2
#    "1,2,1,3"
#    "1,2,2,5"
#    "1,2,4,8"
#    "1,5,2,11"
#    "1,5,4,16"
#    "1,5,8,16"
#    "1,6,4,24"
#    "1,7,4,28"
#    "1,3,1,4"
#    "1,3,2,7"
#    "1,4,2,9"
config_list=(
#    "1,0,0,0"
#    "1,1,1,1"
    "1,1,2,2"
    "1,1,4,4"
    "1,1,8,8"
    "1,1,16,16"
    "1,2,2,2"
    "1,2,2,4"
    "1,2,4,4"
    "1,2,4,8"
    "1,2,4,16"
    "1,2,8,8"
    "1,2,8,16"
    "1,2,8,32"
    "1,2,12,16"
    "1,2,12,32"
    "1,3,4,4"
    "1,3,4,8"
    "1,3,4,16"
    "1,3,8,8"
    "1,3,8,32"
    "1,3,12,16"
    "1,3,12,32"
    "1,4,4,16"
    "1,4,6,32"
    "1,5,4,16"
    "1,5,6,32"
    "1,6,4,16"
    "1,6,6,32"
    "1,7,4,16"
    "1,7,6,32"
    "1,8,4,16"
    "1,8,6,32"
)

TP=4
epoch_num=$(basename "$draft_model_path")

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmarks/bench_model_speedup.py \
    --model-path $target_model_path \
    --speculative-draft-model-path $draft_model_path \
    --port 8000 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --tp-size $TP \
    --config-list "${config_list[@]}" \
    --benchmark-list frontier_reporter:8 \
    --output v1_cuda_graph/long_tp${TP}_batch1.jsonl
