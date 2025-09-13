use crate::config::Config;
use crate::models::{Item, Product};
use crate::utils::file::compute_sha1;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::info;
use reqwest::Client;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::File as AsyncFile;
use tokio::io::AsyncWriteExt;

/// Fetches product information from the API
///
/// # Arguments
/// * `client` - HTTP client instance
/// * `config` - Application configuration
///
/// # Returns
/// Product information structure
pub async fn fetch_product(client: &Client, config: &Config) -> Result<Product> {
    let api_url = format!("{}/products/{}", config.base_url, config.product_id);
    let response = client
        .get(&api_url)
        .send()
        .await
        .context("Failed to fetch product data")?;
    let product: Product = response
        .json()
        .await
        .context("Failed to parse product JSON")?;
    Ok(product)
}

/// Parses human-readable file size string into bytes
///
/// # Arguments
/// * `size_str` - String containing size and unit (e.g., "1.5 MB")
///
/// # Returns
/// Size in bytes as u64
fn parse_file_size(size_str: &str) -> Result<u64> {
    let parts: Vec<&str> = size_str.split_whitespace().collect();
    if parts.len() != 2 {
        bail!("Invalid file size format: {}", size_str);
    }
    let num: f64 = parts[0].parse().context("Failed to parse size number")?;
    let unit = parts[1].to_uppercase();
    let bytes = match unit.as_str() {
        "B" => num as u64,
        "KB" => (num * 1024.0) as u64,
        "MB" => (num * 1024.0 * 1024.0) as u64,
        "GB" => (num * 1024.0 * 1024.0 * 1024.0) as u64,
        _ => bail!("Unknown unit: {}", unit),
    };
    Ok(bytes)
}

/// Downloads a single item with progress tracking and checksum verification
///
/// # Arguments
/// * `client` - HTTP client instance
/// * `config` - Application configuration
/// * `delivery_id` - ID of the delivery containing this item
/// * `item` - Item to download
/// * `download_dir` - Directory to save the downloaded file
/// * `multi_progress` - MultiProgress instance for tracking progress
pub async fn download_item(
    client: Arc<Client>,
    config: Arc<Config>,
    delivery_id: u32,
    item: Item,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    let file_path = format!("{}/{}", config.download_dir, item.item_name);

    if Path::new(&file_path).exists() {
        info!("File already exists: {}. Skipping.", item.item_name);
        return Ok(());
    }

    let download_url = format!(
        "{}/{}",
        config.base_url,
        format!(
            "products/{}/delivery/{}/item/{}/download",
            config.product_id, delivery_id, item.item_id
        )
    );

    info!(
        "Starting download: {} ({}) from {}",
        item.item_name, item.file_size, download_url
    );

    let total_bytes = parse_file_size(&item.file_size)?;

    let pb = multi_progress.add(ProgressBar::new(total_bytes));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(item.item_name.clone());

    let response = client
        .get(&download_url)
        .send()
        .await
        .context("Failed to send download request")?;

    let mut stream = response.bytes_stream();
    let mut file = AsyncFile::create(&file_path)
        .await
        .context("Failed to create file")?;
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush().await.context("Failed to flush file")?;
    pb.finish_with_message(format!("Downloaded: {}", item.item_name));
    info!("Completed download: {}", item.item_name);

    let computed_checksum = compute_sha1(&file_path)?;
    if computed_checksum.to_uppercase() != item.file_checksum.to_uppercase() {
        tokio::fs::remove_file(&file_path)
            .await
            .context("Failed to remove corrupted download")?;
        bail!(
            "Checksum mismatch for {}.\nExpected: {}\nActual: {}",
            item.item_name,
            item.file_checksum,
            computed_checksum
        );
    }

    info!("Checksum verified for {}", item.item_name);
    Ok(())
}
