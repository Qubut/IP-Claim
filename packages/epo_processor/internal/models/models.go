package models

// Product is a top-level entry in the EPO bulk-data catalogue (e.g.
// "EP full-text data"). One product groups many [Delivery] shipments.
type Product struct {
	ID         uint32     `json:"id"`
	Name       string     `json:"name"`
	Deliveries []Delivery `json:"deliveries"`
}

// Delivery is a periodic shipment within a [Product] (typically weekly).
// Each delivery aggregates one or more downloadable [Item] archives.
type Delivery struct {
	DeliveryID             uint32 `json:"deliveryId"`
	DeliveryName           string `json:"deliveryName"`
	DeliveryExpiryDatetime string `json:"deliveryExpiryDatetime,omitempty"`
	Items                  []Item `json:"items"`
}

// Item is a single downloadable archive (typically a .tar or .zip)
// within a [Delivery]. FileChecksum is SHA-1, hex-encoded; case is
// catalogue-dependent and must be compared case-insensitively
// (see RFC 4648 §8).
type Item struct {
	ItemID                  uint32 `json:"itemId"`
	ItemName                string `json:"itemName"`
	FileSize                string `json:"fileSize"`
	FileChecksum            string `json:"fileChecksum"`
	ItemPublicationDatetime string `json:"itemPublicationDatetime"`
}
