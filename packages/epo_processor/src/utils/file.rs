use anyhow::{Context, Result};
use sha1::{Digest, Sha1};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Creates directory if it doesn't exist
///
/// # Arguments
/// * `dir` - Path to the directory
pub fn create_dir(dir: &str) -> Result<()> {
    if !Path::new(dir).exists() {
        std::fs::create_dir(dir).context(format!("Failed to create directory {}", dir))?;
    }
    Ok(())
}

/// Computes the SHA-1 checksum of a file
///
/// # Arguments
/// * `file_path` - Path to the file to compute checksum for
///
/// # Returns
/// Hexadecimal string representation of the SHA-1 hash
pub fn compute_sha1(file_path: &str) -> Result<String> {
    let mut hasher = Sha1::new();
    let mut file = File::open(file_path)?;
    let mut buffer = [0; 4096]; // 4KB buffer

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();

    Ok(format!("{:X}", result))
}
