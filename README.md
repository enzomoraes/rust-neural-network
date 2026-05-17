# rust-neural-network

A high-performance, dependency-free neural network library in Rust designed to demystify machine learning fundamentals while leveraging Rust's performance benefits.

## Folders

### `neural-network/` (Library)
The core neural network library containing the foundational components for building and training neural networks. Includes layer implementations (dense layers), activation functions (ReLU, sigmoid, tanh, softmax), loss functions (MSE, cross-entropy), and serialization support for saving/loading trained models.

### `xor/` (Example Application)
A simple demonstration application that trains a neural network to solve the XOR problem. This is a great starting point to understand how to use the library and see a complete training pipeline in action.

### `mnist/` (Example Application)
A digit classification application trained on the MNIST dataset. Demonstrates training on a realistic machine learning problem with larger datasets and model persistence. This example shows how to build and evaluate models on real-world data.

#### Data Augmentation
The MNIST training pipeline includes data augmentation to improve model generalization and robustness:
- **Gaussian Noise**: Adds random Gaussian noise (σ = 0.05) to training images
- **Image Rotation**: Rotates images by ±10 degrees to simulate different digit orientations
- **Combined Augmentations**: Applies noise to rotated images for additional diversity

Each training image is augmented to create 6 variants (original, noisy, rotated ±10°, and noisy rotations), effectively multiplying the training set size by 6x. Augmented samples are exported for visualization during training.

### `mnist-webasm/` (WebAssembly Application)
A WebAssembly version of the MNIST model compiled for browser execution. Enables running trained neural networks directly in the browser without a server backend, showcasing the library's portability.

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork and Clone**: Fork the repository and clone your fork locally.

2. **Make Changes**: 
   - Create a feature branch for your changes (`git checkout -b feature/your-feature`)
   - Ensure your code maintains the no-dependencies philosophy where possible
   - Follow Rust conventions and best practices

### Guidelines
- Keep the library free of external dependencies where possible
- Maintain readable, well-documented code
- Ensure changes align with the goal of demystifying machine learning concepts
- Performance optimizations should not sacrifice code clarity

### Roadmap
- MNIST
  - Add a 11º class as unknown.