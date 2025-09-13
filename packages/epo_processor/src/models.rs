use serde::Deserialize;

/// Product information structure
#[derive(Deserialize, Debug)]
pub struct Product {
    /// Unique product identifier
    pub id: u32,
    /// Human-readable product name
    pub name: String,
    /// Product description
    pub description: String,
    /// List of available deliveries for this product
    pub deliveries: Vec<Delivery>,
}

/// Delivery information structure
#[derive(Deserialize, Debug)]
pub struct Delivery {
    /// Unique delivery identifier
    #[serde(rename = "deliveryId")]
    pub delivery_id: u32,
    /// Human-readable delivery name
    #[serde(rename = "deliveryName")]
    pub delivery_name: String,
    /// Publication timestamp of the delivery
    #[serde(rename = "deliveryPublicationDatetime")]
    pub delivery_publication_datetime: String,
    /// Optional expiration timestamp of the delivery
    #[serde(rename = "deliveryExpiryDatetime")]
    pub delivery_expiry_datetime: Option<String>,
    /// List of items in this delivery
    pub items: Vec<Item>,
}

/// Item information structure
#[derive(Deserialize, Debug, Clone)]
pub struct Item {
    /// Unique item identifier
    #[serde(rename = "itemId")]
    pub item_id: u32,
    /// File name of the item
    #[serde(rename = "itemName")]
    pub item_name: String,
    /// Human-readable file size with unit
    #[serde(rename = "fileSize")]
    pub file_size: String,
    /// SHA-1 checksum of the file
    #[serde(rename = "fileChecksum")]
    pub file_checksum: String,
    /// Publication timestamp of the item
    #[serde(rename = "itemPublicationDatetime")]
    pub item_publication_datetime: String,
}
