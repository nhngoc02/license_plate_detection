# YOLO License Plate Detection Project

This project involves the training, evaluation, and fine-tuning of YOLO models for license plate detection using two datasets: the **CCPD** dataset and the **UFPR-ALPR** dataset. The models were trained, validated, and tested under various configurations to explore cross-dataset tuning and the impact of hyperparameter optimization.

## Project Overview

The YOLO models were developed and evaluated in the following configurations:

- **Model 1**:
  - Trained on **200 images** from the CCPD dataset.
  - Validated on **60 images** and tested on **30 images**.
  
- **Model 2**:
  - Trained on **1000 images** from the CCPD dataset.
  - Validated on **300 images** and tested on **200 images**.

- **Model 3**:
  - A refined version of Model 2, with custom hyperparameter optimization.
  - **Model 3.1**: Fine-tuned and tested further using the CCPD dataset.

- **Model 4**:
  - Trained on **1800 images** from the UFPR-ALPR dataset.
  - Validated on **900 images** and tested on **1800 images**.
  - **Model 4.1**: Fine-tuned and tested further using the CCPD dataset to explore cross-dataset performance.

## Results
Summary of Model Accuracy Metrics

| **Model**     | **Train/Test** | **Precision(B)** | **mAP50(B)** | **mAP50-95(B)** |
|---------------|----------------|------------------|--------------|-----------------|
| **Model 1**   | Test           | 1.0              | 0.9559       | 0.636           |
|               | Test           | 1.0              | 0.995        | 0.7005          |
| **Model 2**   | Train          | 0.9992           | 0.995        | 0.7235          |
|               | Test           | 0.999            | 0.995        | 0.7324          |
| **Model 3**   | Train          | 0.9996           | 0.995        | 0.7499          |
|               | Test           | 1.0              | 0.995        | 0.729           |
| **Model 3.1** | Train          | 0.9775           | 0.97341      | 0.6326          |
|               | Test           | 0.37728          | 0.12689      | 0.0578          |
| **Model 4**   | Train          | 0.9756           | 0.99365      | 0.6899          |
|               | Test           | 0.96268          | 0.98187      | 0.70983         |
| **Model 4.1** | Train          | 1.0              | 0.995        | 0.70179         |
|               | Test           | 0.9992           | 0.995        | 0.6756          |

---

For further details, check out project-report.pdf.

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-name/license-plate-detection.git
cd license-plate-detection
```

### 2. Install Dependencies


### 3. Run the Trained Models
To train a YOLO model or evaluate the trained models, use the provided Jupyter notebooks or scripts:
```python
# Load a YOLO model (e.g., yolov8_ccpd_1k_tuned.pt)
from ultralytics import YOLO

# Load the trained model
loaded_model = YOLO('/license-plate-detection/models/yolov8_ccpd_1k_tuned.pt')

# Validate the model on a test dataset
# Ensure the dataset is in YOLO-compatible format
loaded_model.val(data='/example.yaml', split='test')

# Run inference on a single image
results = loaded_model('path/to/your/image.jpg')

# Display results (e.g., bounding boxes, labels)
results.show()
```

### 4. View Visualizations
The detection visualization script is included in the `/scripts/mics_bounding_box_visualization.ipynb` file.

An example of the model performance:
![Detection Example](https://drive.google.com/uc?export=view&id=1JbSYWsV3m4ggN0BPDU3GV2r4VokC_n88 "YOLO Detection Example")

## Additional Information

For detailed explanations of the models, datasets, training configurations, and results, please refer to the [Project Report](./project_report.pdf).

---

## Datasets

- **CCPD Dataset**: A large-scale dataset for license plate detection.
  ```
  @article{xu2018towards,
    title={Towards end-to-end license plate detection and recognition: A large dataset and baseline},
    author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
    journal={arXiv preprint arXiv:1806.10447},
    year={2018}
  }
  ```
- **UFPR-ALPR Dataset**: A dataset designed for automatic license plate recognition.
  ```
  @article{silva2018license,
    title={License plate detection and recognition in unconstrained scenarios},
    author={Silva, Sergio and Jung, Claudio},
    journal={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2018}
  }
  ```

You can find more details about how these datasets were used in the [Project Report](./project_report.pdf).
