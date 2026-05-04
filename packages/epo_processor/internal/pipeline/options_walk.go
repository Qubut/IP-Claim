package pipeline

import "io"

// WalkConfig parameterises the archive walker. All fields are optional;
// zero values give the defaults documented per field.
type WalkConfig struct {
	// SpoolDir is where zip archives are spooled (zip needs random access).
	// Tar / tar.gz never use it. Defaults to OS temp.
	SpoolDir string

	// EntrySelector decides whether an entry is emitted to the extractor.
	// Nested archives are always recursed regardless. Defaults to IsXML.
	EntrySelector func(name string) bool

	// KeepExtracted, when true, tees each selected entry's bytes to a
	// file under ExtractedDir while the consumer reads it.
	KeepExtracted bool

	// ExtractedDir receives kept entry files. Required when KeepExtracted is true.
	ExtractedDir string
}

func (w WalkConfig) selector() func(string) bool {
	if w.EntrySelector != nil {
		return w.EntrySelector
	}
	return IsXML
}

// teeIfKeep returns r unchanged when KeepExtracted is false; otherwise
// it tees r to a file under ExtractedDir. The returned closer must be
// invoked when the consumer is done.
func (w WalkConfig) teeIfKeep(archive, entry string, r io.Reader) (io.Reader, func() error, error) {
	return optionalTee(w.KeepExtracted, archive, entry, w.ExtractedDir, r)
}
