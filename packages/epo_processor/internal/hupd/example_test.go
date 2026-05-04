package hupd_test

import (
	"fmt"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/hupd"
)

func ExampleNormalizeUSID() {
	// Strips the "US" prefix and kind code to yield the bare numeric string
	// that matches the EPO patent_id after the same normalisation.
	fmt.Println(hupd.NormalizeUSID("US9114971B2"))
	fmt.Println(hupd.NormalizeUSID("US20120043352A1"))
	fmt.Println(hupd.NormalizeUSID("EP1234567A1")) // non-US → empty
	// Output:
	// 9114971
	// 20120043352
	//
}

func ExampleNormalizeHUPDPatentNumber() {
	// HUPD stores granted patent numbers as plain numeric strings.
	// Normalisation trims whitespace; the string is returned unchanged
	// if it is already a bare number.
	fmt.Println(hupd.NormalizeHUPDPatentNumber("  9114971  "))
	fmt.Println(hupd.NormalizeHUPDPatentNumber("9114971"))
	// Output:
	// 9114971
	// 9114971
}

func ExampleNormalizeFamily() {
	// Filters a family-member list to US-normalised numeric IDs only.
	// Non-US identifiers and unparseable entries are silently dropped.
	ids := hupd.NormalizeFamily([]string{
		"US9114971B2",
		"EP1234567A1", // non-US, dropped
		"US20120043352A1",
		"", // empty, dropped
	})
	for _, id := range ids {
		fmt.Println(id)
	}
	// Output:
	// 9114971
	// 20120043352
}
