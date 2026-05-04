package pipeline

import (
	"path/filepath"
	"strings"
)

// DetectKind returns the ArchiveKind inferred from name's extension.
func DetectKind(name string) ArchiveKind {
	lower := strings.ToLower(filepath.Base(name))
	switch {
	case strings.HasSuffix(lower, ".tar.gz"), strings.HasSuffix(lower, ".tgz"):
		return KindTarGz
	case strings.HasSuffix(lower, ".tar"):
		return KindTar
	case strings.HasSuffix(lower, ".zip"):
		return KindZip
	default:
		return KindUnknown
	}
}

// IsXML returns true for entries that should be parsed.
func IsXML(name string) bool {
	return strings.EqualFold(filepath.Ext(name), ".xml")
}
