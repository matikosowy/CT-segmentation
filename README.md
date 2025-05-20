# CT-segmentation

A project for automatic segmentation of internal organs from Computed Tomography (CT) images using deep neural networks.

## Project Goal

The main purpose of this project is to provide a solution for automatic segmentation of internal organs from CT images. The project enables:

- Multi-organ segmentation (default: left kidney, right kidney, left adrenal gland, right adrenal gland)
- Training two neural network architectures: U-Net and SegResNet
- Calculating segmentation quality metrics (Dice, IoU, Precision, Recall, F1)
- Visualization of segmentation results

## Approach

The project focuses on a 2D segmentation approach, processing CT images slice-by-slice. Key components:

- **Models**: Two architectures implemented: U-Net and SegResNet, using the MONAI library
- **Data Processing**: Automatic loading and preparation of CT data with organ masks
- **Training**: Uses DiceCELoss, Adam optimizer, and automatic early stopping
- **Evaluation**: Calculation of segmentation metrics for each organ and visualization of results

3D segmentation approach in progress.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/matikosowy/CT-segmentation.git
   cd CT-segmentation
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure

Data should be arranged as follows:

```
dataset/
  ├── patient_001/
  │     ├── ct.nii.gz
  │     └── segmentations/
  │           ├── kidney_left.nii.gz
  │           ├── kidney_right.nii.gz
  │           ├── adrenal_gland_left.nii.gz
  │           └── adrenal_gland_right.nii.gz
  ├── patient_002/
  │     ├── ct.nii.gz
  │     └── segmentations/
  ...
```

## Usage

### Training a Model

```bash
python ctseg.py --dataset /path/to/data --model segresnet --epochs 30 --batch_size 16 --run_name my_experiment
```

#### Main Parameters:

- `--dataset`: path to data directory
- `--model`: model architecture (`unet` or `segresnet`)
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--run_name`: experiment name (for organizing results)
- `--target_organs`: list of organs to segment (default: kidney_right kidney_left adrenal_gland_right adrenal_gland_left)
- `--min_organ_pixels`: minimum number of pixels for each organ (default: 50 50 30 30)
- `--mode`: segmentation mode, 2D or 3D (default: "2d")

### Resuming Training

```bash
python ctseg.py --resume --checkpoint /path/to/checkpoint.pth --dataset /path/to/data --epochs 10
```

### Model Evaluation

```bash
python ctseg.py --inference --checkpoint /path/to/checkpoint.pth --dataset /path/to/data
```

### Usage Examples

1. Training SegResNet model on 50 patients:
   ```bash
   python ctseg.py --model segresnet --num_patients 50 --dataset dataset --run_name segresnet_50_patients
   ```

2. Training U-Net model for kidneys only:
   ```bash
   python ctseg.py --model unet --target_organs kidney_left kidney_right --min_organ_pixels 50 50 --dataset dataset
   ```

3. Evaluating a trained model:
   ```bash
   python ctseg.py --inference --checkpoint runs/[model_name].pth --dataset dataset
   ```

## Results

Training and evaluation results are saved in the `runs/[experiment_name]` directory, where you'll find:

- `best_model.pth`: the best model according to Dice Score
- `train_plot.png`: graph of loss function and metrics during training
- `eval/`: directory with evaluation results, including:
  - Segmentation visualizations on the test set
  - `evaluation_metrics.txt` file with detailed metrics for each organ

