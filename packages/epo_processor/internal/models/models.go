package models

type Product struct {
	Id   uint32 `json:"id"`
	Name string `json:"name"`
	// description string // not used
	Deliveries []Delivery `json:"deliveries"`
}

type Delivery struct {
	DeliveryID   uint32 `json:"deliveryId"`
	DeliveryName string `json:"deliveryName"`
	// deliveryPublicationDatetime string // not used
	DeliveryExpiryDatetime string `json:"deliveryExpiryDatetime,omitempty"`
	Items                  []Item `json:"items"`
}

type Item struct {
	ItemId                  uint32 `json:"itemId"`
	ItemName                string `json:"itemName"`
	FileSize                string `json:"fileSize"`
	FileChecksum            string `json:"fileChecksum"`
	ItemPublicationDatetime string `json:"itemPublicationDatetime"`
}
