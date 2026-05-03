CUDA_VISIBLE_DEVICES=4,5,6,7 python text_to_image/playground_fksteering.py \
  --prompt-path text_to_image/prompt_files/test_prompts_4.json \
  --repeat-count 5 \
  --guidance-reward-fns VLMOCRScoreV2 \
  --metrics-to-compute "VLMOCRScoreV2#VLMOCRScore#ImageReward" \
  --lmbda 4.0 \
  --num-particles 8 \
  --resample-frequency 10 \
  --time-steps 100 \
  --potential-type max \
  --resampling-t-start 10 \
  --resampling-t-end 60 \
  --include-terminal-resample \
  --disable-smc

  CUDA_VISIBLE_DEVICES=4,5,6,7 python text_to_image/playground_fksteering.py \
  --prompt-path text_to_image/prompt_files/test_prompts_5.json \
  --repeat-count 5 \
  --guidance-reward-fns VLMOCRScoreV2 \
  --lmbda 4.0 \
  --num-particles 8 \
  --resample-frequency 10 \
  --time-steps 100 \
  --potential-type max \
  --resampling-t-start 10 \
  --resampling-t-end 60 \
  --include-terminal-resample \
  --disable-smc

  CUDA_VISIBLE_DEVICES=0,1,2,3 python text_to_image/playground_fksteering.py \
  --prompt-path text_to_image/prompt_files/color_benchmark.json \
  --repeat-count 10 \
  --guidance-reward-fns ImageReward \
  --lmbda 4.0 \
  --num-particles 8 \
  --resample-frequency 10 \
  --time-steps 100 \
  --potential-type max \
  --resampling-t-start 10 \
  --resampling-t-end 60 \
  --include-terminal-resample

  python3 text_to_image/analyze_vlm_ocr_failures.py \
  --log-path output/20260430-004234/vlm_ocr_intermediate_logs.jsonl \
  --output-dir output/20260430-004234/analysis_vlm_ocr