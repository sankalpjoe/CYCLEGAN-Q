#  CycleGAN-Q

This project implements a Quantum-enhanced CycleGAN architecture for image-to-image translation, specifically designed for day-to-night and night-to-day transformation of images. The implementation leverages quantum computing concepts to potentially improve the quality and diversity of generated images.

## Overview

CycleGAN is a powerful image-to-image translation model that can learn to transform images from one domain to another without paired training examples. This quantum-enhanced version extends the classical CycleGAN by incorporating quantum computing elements through Qiskit, providing a novel approach to generative adversarial networks.

## Key Features

- **Quantum Circuit Integration**: Uses Qiskit to implement quantum circuits that participate in the image generation process
- **Hybrid Architecture**: Combines classical convolutional layers with quantum layers for optimal performance
- **Memory Efficient**: Implements strategies to manage GPU memory efficiently when running complex quantum operations
- **Mixed Precision Training**: Uses half-precision (FP16) during training for faster computation
- **Gradient Accumulation**: Implements gradient accumulation for stable training with quantum components

## Components

### 1. Quantum Circuits (`quantum_circuit.py`)

This module implements the quantum components of the model:

- `QuantumCircuit_Module`: A wrapper around Qiskit's quantum circuit for integration with PyTorch
- `QuantumLayer`: A PyTorch layer that processes feature vectors through a quantum circuit
- `Quantum2DLayer`: A 2D version of the quantum layer for processing image data with convolutional semantics

### 2. Generator Architecture (`quantum_generator.py`)

The quantum-enhanced generator architecture consists of:

- `QuantumConvBlock`: Convolutional blocks with optional quantum enhancement
- `QuantumResidualBlock`: Residual blocks with quantum components
- `QuantumGenerator`: A full generator model that integrates quantum layers at strategic points

### 3. Training Script (`quantum_train.py`)

A modified training script that handles the quantum-enhanced models:

- Initializes quantum generators and classical discriminators
- Implements the CycleGAN training procedure with quantum components
- Manages GPU memory efficiently for quantum operations
- Saves checkpoints for model evaluation

### 4. Testing Script (`test_quantum_cyclegan.py`)

A script to test the trained models:

- Single image transformation testing
- Comparison between quantum and classical models
- Visualization of results

## Installation Requirements

```bash
pip install torch torchvision albumentations qiskit matplotlib pillow numpy tqdm
```

## Usage

### Training

To train the Quantum CycleGAN:

```bash
python quantum_train.py
```

This will train the model using the configuration settings in `config.py`.

### Testing

To test the model on a single image:

```bash
python test_quantum_cyclegan.py path/to/image.jpg
```

To specify a day image input (default is night):

```bash
python test_quantum_cyclegan.py path/to/image.jpg --day
```

To compare quantum and classical models:

```bash
python test_quantum_cyclegan.py path/to/image.jpg --compare
```

## Model Architecture Details

### Quantum Circuit Design

The quantum circuit used in this implementation:

1. **Data Encoding**: Encodes classical data into quantum states using amplitude encoding
2. **Variational Layers**: Applies parametrized rotation gates (RX, RY, RZ) to qubits
3. **Entanglement**: Creates entanglement between qubits using CNOT gates
4. **Measurement**: Measures the quantum state to generate classical output

### Hybrid Processing

The model uses a hybrid approach where:

1. Classical convolutional layers extract features from images
2. Selected features are processed through quantum circuits
3. Quantum and classical outputs are blended for the final result

This design allows the model to benefit from quantum processing while maintaining computational efficiency.

## Performance Considerations

- **Memory Usage**: Quantum simulation requires significant memory; the implementation uses strategies like downsampling and selective quantum layer application
- **Training Time**: Training with quantum components is computationally intensive; a full training run may take significantly longer than classical CycleGAN
- **Batch Size**: The default batch size is reduced to accommodate quantum operations

## Experimental Results

Based on experimental observations, the quantum-enhanced CycleGAN may provide:

1. **Improved Detail**: Enhanced preservation of fine details in some cases
2. **Different Color Distributions**: Potentially more diverse color transformations
3. **Trade-offs**: Some computational overhead compared to classical versions

## Future Directions

- Integration with quantum hardware through Qiskit for real quantum advantage
- Parameter optimization strategies specific to quantum circuits
- Exploration of different quantum circuit architectures for image generation

## References

- Original CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
- Qiskit documentation: [Qiskit.org](https://qiskit.org/)
- PyTorch documentation: [PyTorch.org](https://pytorch.org/)

## License

This project is open source and available under the MIT License.
