SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

target_model_path="/mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/hz/group_airesearch_1/users/juqiuwang/models/hf_iterr_666"
#draft_model_path=${ROOT_DIR}/outputs/QwQ-32B-eagle3/epoch_9
# draft_model_path=${ROOT_DIR}/outputs/QwQ-32B-eagle3_v1/epoch_0
# draft_model_path=${ROOT_DIR}/outputs2/QwQ-32B-eagle3_v2/epoch_0
draft_model_path="/mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/hz/group_airesearch_1/users/juqiuwang/models/eagle_draft_model_hf_iterr_666"
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
    "1,0,0,0"
    "1,6,4,16"
)

TP=4
epoch_num=$(basename "$draft_model_path")

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../evaluation_script/sglang_eagle3.py \
    --model-path $target_model_path \
    --speculative-draft-model-path $draft_model_path \
    --port 40000 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --tp-size $TP \
    --config-list "${config_list[@]}" \
    --benchmark-list frontier_reporter:8 \
    --output ../../result/eagle_sglang_baseline_cuda_graph/hf666_reporter_tp${TP}_ba1.jsonl 
