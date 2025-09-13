use anyhow::{Context, Result};
use dashmap::DashSet;
use flate2::read::GzDecoder;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{error, info};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tar::Archive;
use walkdir::WalkDir;
use zip::ZipArchive;

/// Recursively extracts all supported archive files (gz, tar, zip) in a directory.
///
/// # Arguments
/// * `directory` - The root directory to search for archive files
/// * `delete_after_extract` - If true, deletes the archive file after successful extraction
/// * `multi_progress` - Shared multi-progress bar for tracking extraction progress
///
/// # Returns
/// Returns `Ok(())` if all archives are processed successfully, or an error if any operation fails.
///
/// # Errors
/// Returns an error if directory traversal fails or any archive extraction fails.
pub fn extract_all_in_directory(
    directory: &str,
    delete_after_extract: bool,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    // Uses a thread-safe set to track processed directories
    let processed_dirs = Arc::new(DashSet::new());

    // First, find all directories that might contain archives (including nested ones)
    let all_dirs: Vec<PathBuf> = WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.path().to_owned())
        .collect();

    all_dirs.par_iter().for_each(|dir| {
        if processed_dirs.contains(dir) {
            return;
        }

        processed_dirs.insert(dir.clone());

        if let Err(e) = process_directory(dir, delete_after_extract, multi_progress.clone()) {
            error!("Failed to process directory {}: {}", dir.display(), e);
        }
    });

    Ok(())
}

/// Processes a single directory, extracts archives, and returns subdirectories
fn process_directory(
    directory: &Path,
    delete_after_extract: bool,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    // Find all archive files in this directory
    let archive_files: Vec<PathBuf> = match fs::read_dir(directory) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext, "gz" | "zip" | "tar"))
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect(),
        Err(e) => {
            error!("Failed to read directory {}: {}", directory.display(), e);
            return Ok(());
        }
    };

    if archive_files.is_empty() {
        return Ok(());
    }

    let dir_progress = multi_progress.add(ProgressBar::new(archive_files.len() as u64));
    dir_progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    dir_progress.set_message(format!("Extracting archives in: {}", directory.display()));

    // Extract all archive files in this directory
    archive_files.par_iter().for_each(|file_path| {
        if let Some(file_str) = file_path.to_str() {
            if let Err(e) = extract_file(file_str, delete_after_extract, multi_progress.clone()) {
                error!("Failed to extract {}: {}", file_str, e);
            }
        }
        dir_progress.inc(1);
    });

    dir_progress.finish_and_clear();
    Ok(())
}

/// Extracts a single compressed file based on its extension.
///
/// Supports .gz (both gzip-compressed files and tar.gz archives), .tar, and .zip formats.
///
/// # Arguments
/// * `file_path` - Path to the archive file to extract
/// * `delete_after_extract` - If true, deletes the archive file after successful extraction
/// * `multi_progress` - Shared multi-progress bar for tracking extraction progress
///
/// # Returns
/// Returns `Ok(())` if extraction is successful, or an error if extraction fails.
///
/// # Errors
/// Returns an error if:
/// - The file format is unsupported
/// - File operations (reading/writing/deleting) fail
/// - The archive is corrupt or invalid
pub fn extract_file(
    file_path: &str,
    delete_after_extract: bool,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    let path = Path::new(file_path);
    let parent_dir = path
        .parent()
        .context("Failed to get parent directory")?
        .to_str()
        .context("Failed to convert path to string")?;

    let file_stem = path
        .file_stem()
        .context("Failed to get file stem")?
        .to_str()
        .context("Failed to convert file stem to string")?;

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .context("Failed to get file extension")?;

    let file_progress = multi_progress.add(ProgressBar::new(1));
    file_progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}]")
            .unwrap()
            .progress_chars("#>-"),
    );
    file_progress.set_message(format!(
        "Extracting: {}",
        path.file_name().unwrap().to_string_lossy()
    ));

    info!("Extracting {} to {}", file_path, parent_dir);

    match extension {
        "gz" => {
            if file_stem.ends_with(".tar") {
                extract_tar_gz(file_path, parent_dir)?;
            } else {
                extract_gz(file_path, parent_dir)?;
            }
        }
        "tar" => {
            extract_tar(file_path, parent_dir)?;
        }
        "zip" => {
            extract_zip(file_path, parent_dir, multi_progress.clone())?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported file format: {}", extension));
        }
    }

    if delete_after_extract {
        fs::remove_file(file_path).context("Failed to delete archive after extraction")?;
        info!("Deleted archive: {}", file_path);
    }

    file_progress.finish_and_clear();
    Ok(())
}

fn extract_zip(
    file_path: &str,
    output_dir: &str,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    let file = fs::File::open(file_path)?;
    let mut archive = ZipArchive::new(file)?;

    let zip_progress = multi_progress.add(ProgressBar::new(archive.len() as u64));
    zip_progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    zip_progress.set_message("Extracting ZIP contents");

    (0..archive.len()).try_for_each(|i| -> Result<(), anyhow::Error> {
        let mut file = archive.by_index(i)?;
        let outpath = Path::new(output_dir).join(file.name());

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }

        zip_progress.inc(1);
        zip_progress.set_message(format!("Extracting: {}", file.name()));

        Ok(())
    })?;

    zip_progress.finish_and_clear();
    Ok(())
}

fn extract_tar_gz(file_path: &str, output_dir: &str) -> Result<()> {
    let tar_gz = fs::File::open(file_path)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive
        .unpack(output_dir)
        .context("Failed to unpack tar.gz archive")?;
    Ok(())
}

fn extract_gz(file_path: &str, output_dir: &str) -> Result<()> {
    let gz = fs::File::open(file_path)?;
    let mut decoder = GzDecoder::new(gz);
    let output_path = Path::new(output_dir).join(
        Path::new(file_path)
            .file_stem()
            .context("Failed to get file stem")?,
    );

    let mut output_file = fs::File::create(&output_path)?;
    std::io::copy(&mut decoder, &mut output_file)?;

    Ok(())
}

fn extract_tar(file_path: &str, output_dir: &str) -> Result<()> {
    let file = fs::File::open(file_path)?;
    let mut archive = Archive::new(file);
    archive
        .unpack(output_dir)
        .context("Failed to unpack tar archive")?;
    Ok(())
}
