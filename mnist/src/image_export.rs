use image::{ImageBuffer, Luma};
use std::fs;

const IMAGE_SIZE: u32 = 28;

pub fn save_image(pixels: &[f32], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = ImageBuffer::new(IMAGE_SIZE, IMAGE_SIZE);

    for (idx, &pixel) in pixels.iter().enumerate() {
        let y = (idx as u32) / IMAGE_SIZE;
        let x = (idx as u32) % IMAGE_SIZE;
        let gray = (pixel * 255.0).clamp(0.0, 255.0) as u8;
        img.put_pixel(x, y, Luma([gray]));
    }

    img.save(filename)?;
    Ok(())
}

pub fn save_augmented_samples(
    augmented_images: &[Vec<f32>],
    num_samples: usize,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;

    let sample_count = (num_samples * 6).min(augmented_images.len());

    for (idx, image) in augmented_images.iter().take(sample_count).enumerate() {
        let sample_idx = idx / 6;
        let aug_type = match idx % 6 {
            0 => "01_original",
            1 => "02_noisy",
            2 => "03_rotated_-10",
            3 => "04_rotated_-10_noisy",
            4 => "05_rotated_+10",
            5 => "06_rotated_+10_noisy",
            _ => unreachable!(),
        };

        save_image(
            image,
            &format!("{}/sample_{:03}_{}.png", output_dir, sample_idx, aug_type),
        )?;
    }

    println!(
        "✓ Saved {} augmented samples ({} images) to '{}'",
        sample_count, sample_count, output_dir
    );
    Ok(())
}
