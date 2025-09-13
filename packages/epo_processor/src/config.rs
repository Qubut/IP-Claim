use clap::Parser;
use serde::Deserialize;

/// Command-line arguments for the downloader tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to the configuration file in YAML format
    #[arg(short, long, default_value = "config.yaml")]
    pub config: String,
}

/// Application configuration structure
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    /// Base URL for the API endpoints
    pub base_url: String,
    /// Product ID to download
    pub product_id: u32,
    /// Directory where files will be downloaded
    pub download_dir: String,
    /// Whether to download the files
    #[serde(default = "default_download")]
    pub download: bool,
    /// Whether to extract downloaded archives
    #[serde(default = "default_extract")]
    pub extract: bool,
    /// Whether to parse xml files
    #[serde(default = "default_parse")]
    pub parse: bool,
    /// Path where csv from xml files will be saved
    pub output_csv: String,
    /// No. of XML files to process in parallel
    pub batch_size: usize,
    /// Whether to delete archives after extraction
    #[serde(default = "default_delete_after_extract")]
    pub delete_after_extract: bool,
}

fn default_download() -> bool {
    true
}
fn default_extract() -> bool {
    true
}
fn default_parse() -> bool {
    true
}
fn default_delete_after_extract() -> bool {
    true
}
