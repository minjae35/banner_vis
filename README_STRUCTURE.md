# Banner Visual Analysis Project - Folder Structure

```
/home/intern/banner_vis/
├── 📁 src/                           # Source code directory
│   ├── 📁 utils/                     # Utility functions and helpers
│   │   └── nltk_download.py
│   ├── 📁 backup/                    # Backup files
│   ├── 📁 pipelines/                 # Main processing pipelines for different models
│   │   ├── banner_vlm_qwen25_3B.py
│   │   ├── banner_vlm_qwen25_3B_nodet.py 
│   │   ├── banner_vlm_qwen25_7B.py
│   │   ├── banner_vlm_qwen25.py
│   │   ├── banner_vlm_kanana3b.py
│   │   ├── banner_vlm_smolvlm.py
│   │   ├── banner_vlm_varco.py
│   │   ├── banner_vlm_pipeline.py
│   │   ├── banner_vlm_merge_pipeline.py
│   │   ├── detection_segmentation_warp.py
│   │   ├── detection_segmentation_warp_no_vlm.py
│   │   ├── detection_segmentation_warp_ensemble.py
│   │   ├── segmentation.py
│   │   ├── warp_banner_from_mask.py
│   │   ├── test.jpg
│   │   ├── 📁 seg_test_output/       # Segmentation test outputs
│   │   ├── 📁 LLaVA-NeXT/            # LLaVA-NeXT model files
│   │   └── 📁 ~/                     # Temporary files
│   ├── banner_visualized.py          # Banner visualization script
│   └── run_banner_analysis.sh        # Main analysis runner script
│
├── 📁 data/                          # Data directory
│   ├── 📁 base_data/                 # Base datasets
│   │   ├── 📁 warp_10000/            # Warped banner dataset (10k samples)
│   │   │   ├── minimal_7000.jsonl
│   │   │   └── *.jpg
│   │   ├── 📁 crop_7000/             # Cropped banner dataset (7k samples)
│   │   │   ├── minimal_7000.jsonl
│   │   │   ├── minimal_7000.jsonl.backup
│   │   │   ├── 📁 images/
│   │   │   └── 📁 images_resized/
│   │   ├── 📁 warp_4500/             # Warped banner dataset (4.5k samples)
│   │   │   ├── warp_4500.jsonl
│   │   │   └── 📁 images/
│   │   └── 📁 flat_3500/             # Flat banner dataset (3.5k samples)
│   │       ├── flat_3500.jsonl
│   │       └── 📁 images/
│   ├── 📁 experiments/               # Experimental datasets
│   │   ├── 📁 datasets/
│   │   │   ├── 📁 originals/         # Original datasets
│   │   │   ├── 📁 cot/               # Chain-of-thought datasets
│   │   │   └── 📁 simple_cot/        # Simple chain-of-thought datasets
│   │   ├── 📁 validation_test/       # Validation and test datasets
│   │   │   └── 📁 datasets/
│   │   └── README.md
│   ├── 📁 100Data/                   # Small dataset (100 samples)
│   │   └── 📁 label/
│   │       ├── kanana_batch_100.fixed.jsonl
│   │       └── kanana_batch_100.fixed2.jsonl
│   ├── 📁 konana/                    # Konana-specific data and scripts
│   │   ├── run_banner_pipline.sh
│   │   ├── banner_pipline.py
│   │   ├── konana_data.ipynb
│   │   ├── konana_test.ipynb
│   │   └── result_single.json
│   └── 📁 FlatData/                  # Flat banner data
│       ├── 📁 banner_syn_custom/
│       │   ├── 📁 image_padded_672x672/
│       │   ├── 📁 label/
│       │   ├── 📁 image/
│       │   └── README.md
│       ├── unzip.py
│       └── banner_syn_custom.zip
│
├── 📁 configs/                       # Configuration files
│   ├── tmpv5.json
│   └── det_config.json
│
├── 📁 experiments/                   # Experiment outputs
│   └── 📁 inference/                 # Inference results
│
├── 📁 tools/                         # Utility tools and scripts
│   ├── 📁 scripts/                   # Various scripts
│   │   └── 📁 evaluation/            # Evaluation scripts
│   ├── 📁 data_preparation/          # Data preparation tools
│   ├── 📁 evaluation/                # Evaluation and analysis tools
│   │   ├── compare_checkpoint_results.py
│   │   ├── check_new_experiments.py
│   │   ├── check_validation_test.py
│   │   ├── check_remaining_data.py
│   │   ├── check_by_image_path.py
│   │   ├── check_exact_experiments.py
│   │   ├── check_experiment_composition.py
│   │   ├── debug_data_analysis.py
│   │   ├── check_unique_ids.py
│   │   ├── analyze_experiment_data.py
│   │   ├── verify_fixed_jsonl.py
│   │   ├── check_unique_data.py
│   │   ├── check_type_overlap.py
│   │   ├── check_experiment_overlap.py
│   │   ├── check_no_warp_setup.py
│   │   ├── verify_experiment_data.py
│   │   ├── metrics.py
│   │   ├── compare_models.py
│   │   ├── evaluate_ocr_metrics.py
│   │   ├── check_base_data_consistency.py
│   │   └── check_data_consistency.py
│   ├── 📁 data_processing/           # Data processing tools
│   ├── 📁 model_management/          # Model management tools
│   ├── 📁 testing/                   # Testing tools
│   ├── 📁 utilities/                 # General utilities
│   ├── 📁 batch_processing/          # Batch processing tools
│   ├── eval_ocr_metrics.py
│   └── ocr_evaluation_results.json
│
│
├── 📁 results/                       # Results and outputs
│   ├── 📁 original_model_inference_results/
│   ├── 📁 no_warp_inference_results/
│   ├── 📁 c3f2w1_inference_results/
│   ├── 📁 cw_only_inference_results/
│   └── 📁 bal_equal_inference_results/
│
├── 📁 checkpoints/                   # Model checkpoints and weights
│   ├── DET3.pth
│   ├── DET2.pth
│   ├── ref_cls.ckpt
│   ├── STD_contour.pth
│   ├── yolov5s.pt
│   ├── SEG.ckpt
│   ├── STR.ckpt
│   ├── DET.pt
│   └── STD.pth
│
├── 📁 qwen-tuning/                   # Qwen model fine-tuning
│   └── 📁 Qwen2.5-VL/
│       ├── 📁 qwen-vl-finetune/      # Fine-tuning scripts and configs
│       │   ├── 📁 organized_checkpoints/  # Organized model checkpoints
│       │   │   ├── 📁 bal_equal/          # Balanced equal dataset checkpoints
│       │   │   │   ├── 📁 experiment/
│       │   │   │   └── 📁 simple_experiment/
│       │   │   ├── 📁 c3f2w1/             # C3F2W1 dataset checkpoints
│       │   │   │   ├── 📁 experiment/
│       │   │   │   └── 📁 simple_experiment/
│       │   │   ├── 📁 cw_only/            # CW only dataset checkpoints
│       │   │   │   ├── 📁 experiment/
│       │   │   │   └── 📁 simple_experiment/
│       │   │   └── 📁 no_warp/            # No warp dataset checkpoints
│       │   │       ├── 📁 experiment/
│       │   │       └── 📁 simple_experiment/
│       │   ├── 📁 scripts/               # Training scripts
│       │   │   ├── 📁 experiments/
│       │   │   │   ├── 📁 simple_classification/  # Simple classification experiments
│       │   │   │   │   ├── sft_bal_equal_simple_class.sh
│       │   │   │   │   ├── sft_c3f2w1_simple_class.sh
│       │   │   │   │   ├── sft_cw_only_simple_class.sh
│       │   │   │   │   └── sft_no_warp_simple_class.sh
│       │   │   │   ├── 📁 simple/         # Simple experiments
│       │   │   │   │   ├── run_bal_equal.sh
│       │   │   │   │   ├── run_c3f2w1.sh
│       │   │   │   │   ├── run_cw_only.sh
│       │   │   │   │   └── run_no_warp.sh
│       │   │   │   └── 📁 cot/            # Chain-of-thought experiments
│       │   │   │       ├── run_bal_equal.sh
│       │   │   │       ├── run_c3f2w1.sh
│       │   │   │       ├── run_cw_only.sh
│       │   │   │       └── run_no_warp.sh
│       │   │   ├── 📁 batch_scripts/      # Batch training scripts
│       │   │   │   ├── run_all_experiments.sh
│       │   │   │   ├── run_all_experiments_simple.sh
│       │   │   │   ├── run_all_simple_classification.sh
│       │   │   │   ├── run_all_simple_classification_distributed.sh
│       │   │   │   ├── run_all_simple_classification_large_batch.sh
│       │   │   │   ├── wait_and_run_simple_classification.sh
│       │   │   │   └── wait_and_run_simple_classification_large_batch.sh
│       │   │   ├── 📁 configs/            # Training configurations
│       │   │   │   ├── zero2.json
│       │   │   │   ├── zero3.json
│       │   │   │   ├── zero3_large_batch.json
│       │   │   │   └── zero3_offload.json
│       │   │   ├── 📁 templates/          # Training templates
│       │   │   │   └── training_template.sh
│       │   │   ├── README.md
│       │   │   ├── zero3.json
│       │   │   ├── zero3_offload.json
│       │   │   ├── zero2.json
│       │   │   └── sft.sh
│       │   ├── 📁 qwenvl/                 # QwenVL training code
│       │   │   ├── 📁 data/               # Data processing
│       │   │   │   ├── __init__.py
│       │   │   │   ├── __init__ cot.py
│       │   │   │   ├── data_qwen.py
│       │   │   │   ├── data_qwen_packed.py
│       │   │   │   └── rope2d.py
│       │   │   └── 📁 train/              # Training code
│       │   │       ├── train_qwen.py
│       │   │       ├── trainer.py
│       │   │       ├── argument.py
│       │   │       └── 📁 checkpoints/
│       │   ├── 📁 tools/                  # Training tools
│       │   │   ├── check_image.py
│       │   │   ├── process_bbox.ipynb
│       │   │   └── pack_data.py
│       │   ├── 📁 log/                    # Training logs
│       │   │   └── 📁 cot/
│       │   ├── 📁 demo/                   # Demo files
│       │   └── README.md
│       ├── 📁 web_demo_streaming/         # Web demo with streaming
│       ├── 📁 Qwen2.5-VL/                # Qwen2.5-VL model files
│       ├── 📁 qwen-vl-utils/              # QwenVL utilities
│       │   ├── .python-version
│       │   ├── README.md
│       │   ├── pyproject.toml
│       │   ├── requirements-dev.lock
│       │   ├── requirements.lock
│       │   └── 📁 src/
│       │       └── 📁 qwen_vl_utils/
│       │           ├── __init__.py
│       │           └── vision_process.py
│       ├── 📁 docker/                     # Docker configurations
│       ├── 📁 evaluation/                 # Model evaluation
│       ├── 📁 cookbooks/                  # Usage examples and guides
│       ├── requirements_web_demo.txt
│       ├── web_demo_mm.py
│       ├── .gitignore
│       ├── LICENSE
│       └── README.md
│
├── 📁 json/                          # JSON data files
├── 📁 논문/                          # Paper and research documents
│   └── 📁 tune/
│       └── 📁 24개선된prompt/         # 24 improved prompts
│           └── all_checkpoints_results.json
├── 📁 varco_results/                 # VARCO model results
├── 📁 all_checkpoints_inference_results_original/  # Original checkpoint results
├── 📁 checkpoint_classification_results/           # Classification results
├── 📁 output_backup/                 # Backup outputs
├── 📁 pretrained/                    # Pretrained models
├── 📁 docs/                          # Documentation
├── 📁 __pycache__/                   # Python cache files
├── 📁 ~/                             # Temporary files
│
├── 📄 requirements.txt               # Python dependencies
└── 📄 malgun.ttf                     # Korean font file
```

