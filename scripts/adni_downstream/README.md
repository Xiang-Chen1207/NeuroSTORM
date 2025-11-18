# ADNI Fine-tuning for Alzheimer's Disease Classification

This directory contains scripts and documentation for fine-tuning the NeuroSTORM brain foundation model on the ADNI dataset for Alzheimer's Disease (AD) vs Cognitively Normal (CN) classification.

## Overview

The ADNI fine-tuning implementation includes:
- **Direct .nii.gz loading**: No need to convert to .pt format
- **Automatic 20-frame splitting**: Each 100+ frame fMRI volume is split into consecutive 20-frame segments
- **Label extraction from paths**: Labels (AD/CN) are automatically extracted from file paths
- **Binary classification**: AD (label=1) vs CN (label=0)

## Dataset Structure

Your ADNI data should be organized with three text files containing file paths:

```
/mnt/dataset4/DATASETS/fsl_fmri/adni_split/
├── adni_ad_mni_train.txt
├── adni_ad_mni_val.txt
└── adni_ad_mni_test.txt
```

Each text file contains paths to .nii.gz files, one per line:
```
/mnt/dataset4/DATASETS/fsl_fmri/ADNI(all)/mnispace/cn/ADNI_sub-035S6730_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz_cn_zscore.nii.gz
/mnt/dataset4/DATASETS/fsl_fmri/ADNI(all)/mnispace/ad/ADNI_sub-035S6156_ses-01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz_ad_zscore.nii.gz
...
```

## Data Processing

### NIfTI File Format
- **Expected shape**: (H, W, D, T) where T > 20 (e.g., 96×96×96×100+)
- **Processing**: Each file is split into consecutive 20-frame segments
  - Segment 1: frames 0-19
  - Segment 2: frames 20-39
  - Segment 3: frames 40-59
  - ...
- **Discarded frames**: Any remaining frames that don't form a complete 20-frame segment are dropped

### Label Extraction
Labels are automatically extracted from file paths:
- **AD (label=1)**: Paths containing `/ad/` or `_ad_`
- **CN (label=0)**: Paths containing `/cn/` or `_cn_`

## Files

- **ft_neurostorm_adni_ad_classification.sh**: Main training script for fine-tuning
- **test_adni_dataloader.py**: Test script to verify dataloader functionality
- **README.md**: This documentation

## Quick Start

### 1. Test the Dataloader (Optional but Recommended)

Before starting training, verify that your data can be loaded correctly:

```bash
python scripts/adni_downstream/test_adni_dataloader.py
```

Expected output:
```
================================================================================
Testing ADNI Dataloader
================================================================================

[1/4] Creating fMRIDataModule...
Load dataset ADNI, XXX subjects
  - AD (label=1): XXX files
  - CN (label=0): XXX files

[2/4] Checking dataset sizes...
  Train dataset: XXX samples
  Val dataset: XXX samples
  Test dataset: XXX samples

[3/4] Loading a sample from train dataset...

[4/4] Sample information:
  fMRI sequence shape: torch.Size([1, 96, 96, 96, 20])
  Subject name: ADNI_sub-...
  Target (label): 0 or 1
  Sex: 0
  TR (start frame): 0, 20, 40, ...

✓ Shape verification PASSED: (1, 96, 96, 96, 20) == (1, 96, 96, 96, 20)
================================================================================
✓ All tests PASSED!
================================================================================
```

### 2. Download Pre-trained Model

Download the pre-trained NeuroSTORM model:

```bash
# Create output directory
mkdir -p ./output/neurostorm

# Download from Hugging Face or your pre-trained model location
# Place the checkpoint at: ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt
```

### 3. Run Fine-tuning

Run the fine-tuning script:

```bash
# Make the script executable
chmod +x scripts/adni_downstream/ft_neurostorm_adni_ad_classification.sh

# Run with default batch size (4)
bash scripts/adni_downstream/ft_neurostorm_adni_ad_classification.sh

# Or specify custom batch size
bash scripts/adni_downstream/ft_neurostorm_adni_ad_classification.sh 8
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max_epochs` | 30 | Maximum training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--learning_rate` | 5e-5 | Learning rate |
| `--num_workers` | 8 | Number of data loading workers |
| `--model` | neurostorm | Model architecture |
| `--embed_dim` | 36 | Embedding dimension |
| `--depth` | 2 2 6 2 | Depths for each stage |
| `--num_classes` | 2 | Number of classes (AD vs CN) |
| `--sequence_length` | 20 | Number of frames per sequence |
| `--img_size` | 96 96 96 20 | Input image size |

### GPU Configuration

By default, the script uses 4 GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Modify this in the script based on your available GPUs:
- Single GPU: `export CUDA_VISIBLE_DEVICES=0`
- Two GPUs: `export CUDA_VISIBLE_DEVICES=0,1`
- etc.

Also adjust the training strategy:
- Single GPU: `--strategy gpu` (or remove `--strategy`)
- Multiple GPUs: `--strategy ddp`

## Monitoring Training

Training logs are saved using TensorBoard:

```bash
# View training progress
tensorboard --logdir ./lightning_logs
```

Open your browser and navigate to `http://localhost:6006` to view:
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule
- etc.

## Output

Training outputs are saved in:
```
./lightning_logs/adni_ft_neurostorm_ad_classification/
├── version_0/
│   ├── checkpoints/       # Model checkpoints
│   ├── events.out.tfevents.*  # TensorBoard logs
│   └── hparams.yaml       # Hyperparameters
```

## Evaluation

After training, evaluate the model on the test set:

```bash
python main.py \
  --test_only \
  --test_ckpt_path ./lightning_logs/adni_ft_neurostorm_ad_classification/version_0/checkpoints/best_model.ckpt \
  --dataset_name ADNI \
  --image_path /mnt/dataset4/DATASETS/fsl_fmri/adni_split \
  --downstream_task_type classification \
  --num_classes 2 \
  --task_name diagnosis \
  --model neurostorm \
  --embed_dim 36 \
  --depth 2 2 6 2 \
  --sequence_length 20 \
  --img_size 96 96 96 20
```

## Troubleshooting

### Issue: Out of Memory (OOM) Error

**Solution**: Reduce batch size
```bash
bash scripts/adni_downstream/ft_neurostorm_adni_ad_classification.sh 2
```

### Issue: "Could not extract label from path"

**Solution**: Ensure your file paths contain `/ad/` or `/cn/` to indicate the class.

### Issue: NIfTI file has fewer than 20 frames

**Solution**: The ADNI dataset class will skip files with fewer than 20 frames. Check your data preprocessing.

### Issue: "No module named 'nibabel'"

**Solution**: Install nibabel
```bash
pip install nibabel
```

### Issue: Data loading is slow

**Solution**: Increase number of workers (if you have sufficient CPU cores)
```bash
# In the training script, modify:
--num_workers 16  # Increase based on your CPU cores
```

## Implementation Details

### Modified Files

The following files were modified/created to support ADNI fine-tuning:

1. **datasets/fmri_datasets.py**: Added `ADNI` dataset class
   - Implements direct .nii.gz loading
   - Handles 20-frame splitting
   - Overrides `load_sequence()` method

2. **utils/data_module.py**: Added ADNI support
   - Added `ADNI` to imports
   - Added ADNI case in `get_dataset()`
   - Added ADNI data loading logic in `make_subject_dict()`

3. **main.py**: Added ADNI to dataset choices
   - Updated `--dataset_name` choices to include "ADNI"

4. **scripts/adni_downstream/**: New directory with training scripts
   - Training script: `ft_neurostorm_adni_ad_classification.sh`
   - Test script: `test_adni_dataloader.py`
   - Documentation: `README.md`

### Key Differences from Other Datasets

| Aspect | Other Datasets | ADNI Dataset |
|--------|---------------|--------------|
| Storage format | .pt files (preprocessed) | .nii.gz files (raw) |
| Data structure | `img/` directory with subjects | Text files with file paths |
| Frame extraction | Pre-split into frames | On-the-fly splitting |
| Label source | CSV metadata | File path pattern matching |
| Memory usage | Lower (pre-processed) | Higher (loads full volume) |

## Custom Configuration

To customize the training, edit `ft_neurostorm_adni_ad_classification.sh`:

### Change data paths:
```bash
--image_path /your/custom/path/to/adni_split
```

### Adjust learning rate:
```bash
--learning_rate 1e-4  # Higher learning rate
```

### Freeze feature extractor (linear probing):
```bash
--freeze_feature_extractor
```

### Change optimizer or scheduler:
See `models/lightning_model.py` for available options.

## Citation

If you use this implementation, please cite the NeuroSTORM paper:

```
@article{neurostorm2024,
  title={NeuroSTORM: A Brain Foundation Model for fMRI Analysis},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For questions or issues specific to the ADNI implementation, please open an issue on GitHub.
