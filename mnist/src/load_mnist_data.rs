use std::fmt;
use std::fs::File;
use std::io::{Read, Result};

pub struct MNIST {
    pub train_images: Vec<Vec<f64>>,
    pub train_labels: Vec<Vec<f64>>,
    pub test_images: Vec<Vec<f64>>,
    pub test_labels: Vec<Vec<f64>>,
}

pub struct Image {
    rows: u32,
    columns: u32,
    pub pixels: Vec<f64>,
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = String::new();

        result.push_str(&format!("Image ({} x {})\n", self.rows, self.columns));

        for i in 0..(self.rows * self.columns) {
            if i % self.columns == 0 {
                result.push('\n');
            }
            result.push(match (self.pixels[i as usize] * 255.0) as u8 {
                0 => ' ', // Background (white)
                255 => '█', // Foreground (black)
                _ => '#',   // Other values
            });
        }

        write!(f, "{}", result)
    }
}

pub struct Label {
    pub label: u8,
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Label: {}", self.label)
    }
}

fn read_u32_from_file(file: &mut File) -> Result<u32> {
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn read_u8_from_file(file: &mut File) -> Result<u8> {
    let mut buffer = [0u8; 1];
    file.read_exact(&mut buffer)?;
    Ok(buffer[0])
}

fn read_images(filename: &str) -> Result<Vec<Image>> {
    let mut file = File::open(filename)?;
    let _magic_number = read_u32_from_file(&mut file)?;
    let num_images = read_u32_from_file(&mut file)?;
    let num_rows = read_u32_from_file(&mut file)?;
    let num_columns = read_u32_from_file(&mut file)?;

    let mut images = Vec::with_capacity(num_images as usize);

    for _ in 0..num_images {
        let mut pixels = vec![0u8; (num_rows * num_columns) as usize];
        file.read_exact(&mut pixels)?;
        images.push(Image {
            rows: num_rows,
            columns: num_columns,
            pixels: pixels.iter().map(|p| f64::from(*p) / 255.0).collect(),
        });
    }

    Ok(images)
}

fn read_labels(filename: &str) -> Result<Vec<Label>> {
    let mut file = File::open(filename)?;
    let _magic_number = read_u32_from_file(&mut file)?;
    let num_labels = read_u32_from_file(&mut file)?;

    let mut labels = Vec::with_capacity(num_labels as usize);

    for _ in 0..num_labels {
        let label = read_u8_from_file(&mut file)?;
        labels.push(Label { label });
    }

    Ok(labels)
}

pub fn load_data(directory: &str) -> MNIST {
    let train_images = read_images(format!("{}/train-images.idx3-ubyte", directory).as_str());
    let train_labels = read_labels(format!("{}/train-labels.idx1-ubyte", directory).as_str());
    let test_images = read_images(format!("{}/t10k-images.idx3-ubyte", directory).as_str());
    // let test_images2 = read_images(format!("{}/t10k-images.idx3-ubyte", directory).as_str());

    let test_labels: std::result::Result<Vec<Label>, std::io::Error> =
        read_labels(format!("{}/t10k-labels.idx1-ubyte", directory).as_str());
    // let test_labels2: std::result::Result<Vec<Label>, std::io::Error> = read_labels(format!("{}/t10k-labels.idx1-ubyte", directory).as_str());

    // println!("{}", test_images2.unwrap()[1000]);
    // println!("{}", test_labels2.unwrap()[1000]);

    return MNIST {
        train_images: train_images
            .unwrap()
            .iter()
            .take(1000)
            .map(|image| image.pixels.clone())
            .collect(),
        test_images: test_images
            .unwrap()
            .iter()
            .take(1000)
            .map(|image| image.pixels.clone())
            .collect(),
        train_labels: train_labels
            .unwrap()
            .iter()
            .take(1000)
            .map(|label| {
                let mut inner_vec = vec![0.0; 10];
                inner_vec[label.label as usize] = 1.0 as f64;
                inner_vec
            })
            .collect(),
        test_labels: test_labels
            .unwrap()
            .iter()
            .take(1000)
            .map(|label| {
                let mut inner_vec = vec![0.0; 10];
                inner_vec[label.label as usize] = 1.0 as f64;
                inner_vec
            })
            .collect(),
    };
}
