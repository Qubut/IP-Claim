package typing

type Unit struct{}

type ProgressUpdate struct {
	Action string
	Value  interface{}
}

type ProgressChan chan ProgressUpdate
