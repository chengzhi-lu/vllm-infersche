##########################################
#Reproduce results in Table 3
##########################################

##########################################
#Learning to Rank
##########################################

#Lmsys/LTR/llama-70b Tau=0.62
# python3 trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

#ShareGPT/LTR/llama-70b Tau=0.55
# python3 trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE
# python3 trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama2-70b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

# python3 trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama2-13b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

# python3 trainer.py --config configs/config_prefill_opt_350m.txt --file /root/vllm/benchmarks/output-alpaca-Llama-2-70b-chat-hf.jsonl --job-dir MODEL --run-id opt-350m-llama2-70b-alpaca-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

# python3 trainer.py --config configs/config_prefill_opt.txt --file /root/vllm/examples/analysis/result_analysis/eos_prob_analysis/data/eos_result/output-alpaca-Llama-2-13b-chat-hf.jsonl --job-dir MODEL --run-id opt-125m-llama2-13b-alpaca-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

# python3 trainer.py --config configs/config_prefill_opt.txt --file /root/vllm/benchmarks/output_dir/output-sharegpt-Llama-2-13b-chat-hf.jsonl --job-dir MODEL --run-id opt-125m-llama2-13b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE
python3 trainer.py --config configs/config_prefill_opt_350m.txt --file /root/vllm/examples/analysis/result_analysis/eos_prob_analysis/data/eos_result/output-alpaca-Llama-2-70b-chat-hf.jsonl --job-dir MODEL --run-id opt-350m-llama2-70b-alpaca-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

# python3 trainer.py --config configs/config_prefill_opt.txt --file /root/vllm/benchmarks/output-lmsys-Llama-2-13b-chat-hf.jsonl --job-dir MODEL --run-id opt-350m-llama2-13b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE
# python3 trainer.py --config configs/config_prefill_opt.txt --file /root/vllm/benchmarks/output-lmsys-Llama-2-70b-chat-hf.jsonl --job-dir MODEL --run-id opt-350m-llama2-70b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

##########################################
#Classification
##########################################

#Lmsys/class bucket=100/llama-70b Tau=0.58
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket100-b32 --batch-size 32 --label-group-size 100 

# ShareGPT/class bucket=100/llama-70b Tau=0.49
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket100-b32 --batch-size 32 --label-group-size 100 


#Lmsys/class bucket=10/llama-70b Tau=0.57
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket10-b32 --batch-size 32 --label-group-size 10 

#ShareGPT/class bucket=10/llama-70b Tau=0.48
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket10-b32 --batch-size 32 --label-group-size 10 


#Lmsys/class bucket=1/llama-70b Tau=0.52
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket1-b32 --batch-size 32 --label-group-size 1 

#ShareGPT/class bucket=1/llama-70b Tau=0.28
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket1-b32 --batch-size 32 --label-group-size 1 


#Lmsys/class bucket=820/llama-70b Tau=0.21
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 


#ShareGPT/class bucket=820/llama-70b Tau=0.18
# python3 trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 

##########################################
#Example for fine-tuning on 125M models
##########################################

#Lmsys/LTR/llama-8b Tau=0.64
# python3 trainer.py --config configs/config_prefill_opt.txt --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

#ShareGPT/LTR/llama-8b Tau=0.52
# python3 trainer.py --config configs/config_prefill_opt.txt --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE
# python3 trainer.py --config configs/config_prefill_opt.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-125m-llama2-13b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE


#Lmsys/class bucket=820/llama-70b acc: 0.97
# python3 trainer.py --config configs/config_prefill_opt_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 


#ShareGPT/class bucket=820/llama-70b acc: 0.92
# python3 trainer.py --config configs/config_prefill_opt_classify.txt --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 
