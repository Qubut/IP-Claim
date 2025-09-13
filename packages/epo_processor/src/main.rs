//! A CLI tool for Bulk downloading,extracting and parsing of EPO data with progress tracking and checksum verification.
//!
//! It reads a yaml configuration file, fetches product information from the EPO API,
//! and downloads all items concurrently while displaying progress bars.
//! Each downloaded file is verified using SHA-1 checksums to ensure data integrity.
//! The XML files are then to be extracted and parsed into a csv file

mod config;
mod download;
mod epo_parse;
mod extract;
mod models;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use config::{Args, Config};
use download::{download_item, fetch_product};
use futures::future::join_all;
use indicatif::MultiProgress;
use log::{error, info};
use reqwest::Client;
use serde_yaml;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio;
use utils::file::create_dir;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let config_file = File::open(&args.config).context("Failed to open config file")?;
    let reader = BufReader::new(config_file);
    let config: Arc<Config> =
        Arc::new(serde_yaml::from_reader(reader).context("Failed to parse config YAML")?);

    let client = Arc::new(Client::new());

    let product = fetch_product(&client, &config).await?;

    info!("Product: {} - {}", product.id, product.name);
    info!("Number of deliveries: {}", product.deliveries.len());

    create_dir(&config.download_dir)?;

    let multi_progress = Arc::new(MultiProgress::new());

    let tasks: Vec<_> = product
        .deliveries
        .into_iter()
        .flat_map(|delivery| {
            let delivery_id = delivery.delivery_id.clone();
            let client = client.clone();
            let config = config.clone();
            let multi_progress = multi_progress.clone();
            delivery.items.into_iter().filter_map(move |item| {
                let client_clone = Arc::clone(&client);
                let config_clone = Arc::clone(&config);
                let multi_progress_clone = Arc::clone(&multi_progress);
                let delivery_id = delivery_id.clone();
                let file_path = format!("{}/{}", config.download_dir, item.item_name);
                if config.download {
                    Some(tokio::spawn(async move {
                        download_item(
                            client_clone,
                            config_clone,
                            delivery_id,
                            item,
                            multi_progress_clone,
                        )
                        .await
                    }))
                } else {
                    info!(
                        "Skipping Download, the File or it's Directory already exists: {}",
                        file_path
                    );
                    None
                }
            })
        })
        .collect();
    let results = join_all(tasks).await;

    for result in results {
        if let Err(e) = result {
            error!("Task join error: {:?}", e);
        }
    }

    if config.extract {
        info!("Extracting downloaded files...");
        if let Err(e) = extract::extract_all_in_directory(
            &config.download_dir,
            config.delete_after_extract,
            multi_progress,
        ) {
            error!("Error during top-level extraction: {}", e);
        } else {
            info!("Files extracted sucessfully")
        }
    }

    if config.parse {
        info!("Parsing XML files to CSV...");
        if let Err(e) =
            epo_parse::parse_all_to_csv(&config.download_dir, &config.output_csv, config.batch_size)
        {
            error!("Error during parsing: {}", e);
        } else {
            info!("Parsing completed successfully");
        }
    }

    Ok(())
}
