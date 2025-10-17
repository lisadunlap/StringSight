#! /bin/bash
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/open_assistant/open_assistant_results.jsonl --output_dir results/open_assistant --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/self_instruct/self_instruct_results.jsonl --output_dir results/self_instruct --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/vicuna/vicuna_results.jsonl --output_dir results/vicuna --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/koala/koala_results.jsonl --output_dir results/koala --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/grammar_results.jsonl --output_dir results/grammar --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/helm_wildbench_results.jsonl --output_dir results/wildbench --min_cluster_size 15 --run_metrics

python scripts/run_pipeline.py --method single_model --use_wandb --input_file /home/lisabdunlap/StringSight/data/medhelm/aci_bench_results.jsonl --output_dir results/aci_bench --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/helm/helm_bigcodebench_results_processed.jsonl --output_dir results/bigcodebench --min_cluster_size 15 --run_metrics

python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data.jsonl --output_dir results/taubench_retail --system_prompt agent_system_prompt --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/retail_data_incorrect.jsonl --output_dir results/taubench_retail_incorrect_only --system_prompt agent_system_prompt --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/airline_data.jsonl --output_dir results/taubench_airline --system_prompt agent_system_prompt --run_metrics
python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/taubench/airline_data_incorrect.jsonl --output_dir results/taubench_airline_incorrect_only --system_prompt agent_system_prompt --run_metrics

python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_webdev_sbs.jsonl --output_dir results/arena_webdev_sbs --system_prompt webdev_system_prompt_no_examples --min_cluster_size 15 --run_metrics
python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/arena_sbs.jsonl --output_dir results/arena_sbs --system_prompt sbs_w_metrics_system_prompt --min_cluster_size 15 --run_metrics

python scripts/run_pipeline.py --method side_by_side --use_wandb --input_file data/demo_data/call_center_results_new_oai.jsonl --output_dir results/call_center_bug --system_prompt single_model_system_prompt --min_cluster_size 8 --run_metrics


python scripts/run_pipeline.py --method single_model --use_wandb --input_file data/demo_data/airline_data.jsonl --output_dir results/taubench_airline_demo --system_prompt agent_system_prompt