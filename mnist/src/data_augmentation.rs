use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use rand_distr::Normal;

pub struct AugmentationConfig {
    pub noise_stddev: f32,
    pub rotation_angles: Vec<f32>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            noise_stddev: 0.05,
            rotation_angles: vec![-10.0, 10.0],
        }
    }
}
/// Adds Gaussian noise to an image represented as a flat slice of 32-bit floats.
///
/// This function generates random noise based on a Normal (Gaussian) distribution
/// and adds it element-wise to the original image pixels. The resulting pixel
/// values are then clamped to the valid range of **0.0** to **1.0**.
///
/// # Mathematics
/// Let the original image be represented by a vector $A$.
/// We generate a noise vector $Z$ of the same length, where each element $Z_i$
/// is sampled from a Gaussian distribution with mean $\mu = 0$ and a given
/// standard deviation $\sigma$:
/// $$Z_i \sim \mathcal{N}(0, \sigma^2)$$
///
/// The noisy image $R$ is calculated via vector addition:
/// $$R = A + Z$$
///
/// Finally, a clipping function is applied to ensure no pixel exceeds the bounds:
/// $$R_i = \max(0, \min(1, R_i))$$
///
/// # Arguments
/// * `image` - A slice representing the flattened image pixels (expected range **0.0** to **1.0**).
/// * `stddev` - The standard deviation ($\sigma$) of the Gaussian distribution. Higher values mean more noise.
///
/// # Returns
/// A new `Vec<f32>` containing the image data with applied noise and clamped values.
pub fn add_gaussian_noise(image: &[f32], stddev: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, stddev).unwrap();
    let size = image.len();

    let image_arr = Array1::from_vec(image.to_vec());
    let mut noise_arr = Array1::zeros(size);

    for i in 0..size {
        noise_arr[i] = rng.sample(&normal);
    }

    (image_arr + noise_arr).mapv(|x| x.clamp(0.0, 1.0)).to_vec()
}

/// Rotates a square image by a specified angle using inverse mapping.
///
/// This function treats the 1D slice as a 2D square matrix. To avoid aliasing
/// (empty pixels or "holes" in the resulting image), it iterates over the
/// *target* image matrix and uses an inverse rotation matrix to fetch the
/// corresponding color from the *source* image using Nearest-Neighbor interpolation.
///
/// # Mathematics
/// Let $\theta$ be the rotation angle in radians. To rotate around the geometric
/// center of the image $(c_x, c_y)$, we first translate the origin to the center,
/// apply the 2D inverse rotation matrix $R(-\theta)$, and then translate back:
///
/// $$c_x = c_y = \frac{N - 1}{2}$$
///
/// For every pixel coordinate $(x_{\text{new}}, y_{\text{new}})$ in the output image,
/// the source coordinate $(x_{\text{old}}, y_{\text{old}})$ is found by solving:
///
/// $$
/// \begin{bmatrix} x_{\text{old}} \\ y_{\text{old}} \end{bmatrix} =
/// \begin{bmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{bmatrix}
/// \begin{bmatrix} x_{\text{new}} - c_x \\ y_{\text{new}} - c_y \end{bmatrix} +
/// \begin{bmatrix} c_x \\ c_y \end{bmatrix}
/// $$
///
/// Note that $\cos(-\theta) = \cos(\theta)$ and $\sin(-\theta) = -\sin(\theta)$,
/// which produces the specific signs used in the code's algebraic expansion.
/// The resulting floats are then rounded to the nearest integer to find the valid index.
///
/// # Arguments
/// * `pixels` - A flat slice of `f32` representing a square image.
/// * `angle_degrees` - The rotation angle in degrees (clockwise).
///
/// # Returns
/// A `Vec<f32>` containing the rotated image pixels. Out-of-bounds pixels are set to **0.0**.
///
/// # Panics
/// Panics if the total number of elements in `pixels` is not a perfect square
/// (i.e., it cannot form an $N \times N$ matrix).
pub fn rotate_image(pixels: &[f32], angle_degrees: f32) -> Vec<f32> {
    // 1. Calcular as dimensões da matriz (N x N) usando a raiz quadrada de 'size'
    let n = (pixels.len() as f32).sqrt() as usize;

    let img = ArrayView2::from_shape((n, n), pixels)
        .expect("O tamanho do array não forma uma matriz quadrada perfeita.");

    let mut out_img = Array2::<f32>::zeros((n, n));

    let theta = angle_degrees.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let cx = (n as f32 - 1.0) / 2.0;
    let cy = (n as f32 - 1.0) / 2.0;

    for ((y_new, x_new), pixel_mut) in out_img.indexed_iter_mut() {
        let dx = x_new as f32 - cx;
        let dy = y_new as f32 - cy;

        let x_old = dx * cos_t + dy * sin_t + cx;
        let y_old = -dx * sin_t + dy * cos_t + cy;

        let x_idx = x_old.round() as isize;
        let y_idx = y_old.round() as isize;

        if x_idx >= 0 && x_idx < n as isize && y_idx >= 0 && y_idx < n as isize {
            *pixel_mut = img[[y_idx as usize, x_idx as usize]];
        }
    }

    out_img.into_raw_vec()
}

pub fn augment_mnist(
    images: &[Vec<f32>],
    labels: &[Vec<f32>],
    config: &AugmentationConfig,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut augmented_images = Vec::new();
    let mut augmented_labels = Vec::new();

    for (image, label) in images.iter().zip(labels.iter()) {
        augmented_images.push(image.clone());
        augmented_labels.push(label.clone());

        let noisy_image = add_gaussian_noise(image, config.noise_stddev);
        augmented_images.push(noisy_image);
        augmented_labels.push(label.clone());

        for &angle in &config.rotation_angles {
            let rotated = rotate_image(image, angle);
            let noisy_rotated = add_gaussian_noise(&rotated, config.noise_stddev);

            augmented_images.push(rotated);
            augmented_labels.push(label.clone());

            augmented_images.push(noisy_rotated);
            augmented_labels.push(label.clone());
        }
    }

    (augmented_images, augmented_labels)
}
