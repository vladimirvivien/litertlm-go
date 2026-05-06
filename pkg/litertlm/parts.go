package litertlm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Part is one piece of multimodal input to GenerateMulti /
// GenerateMultiStream / GenerateMultiResponse / GenerateDataMulti.
// Construct via Text, Image, ImageWithMime, ImageFromFile, Audio,
// AudioWithMime, or AudioFromFile.
//
// Part deliberately wraps the C-side InputData rather than aliasing
// it: image and audio segments need an end-marker on the C side
// (InputImageEnd / InputAudioEnd), and a Go-side struct lets
// GenerateDataMulti rewrite the schema instruction into a text part
// without round-tripping through C-allocated memory.
type Part struct {
	kind partKind
	text string
	data []byte
	mime string
}

type partKind int

const (
	partText partKind = iota
	partImage
	partAudio
)

// Text returns a text Part wrapping s.
func Text(s string) Part {
	return Part{kind: partText, text: s}
}

// Image returns an image Part wrapping b. The MIME type is left
// empty; pass ImageWithMime when the caller knows the format, or
// ImageFromFile to derive it from a file extension.
func Image(b []byte) Part {
	return Part{kind: partImage, data: b}
}

// ImageWithMime returns an image Part wrapping b with an explicit
// MIME type (e.g. "image/jpeg", "image/png", "image/webp"). The MIME
// type is metadata for caller introspection; the C side accepts the
// raw bytes regardless.
func ImageWithMime(b []byte, mime string) Part {
	return Part{kind: partImage, data: b, mime: mime}
}

// ImageFromFile reads the file at path and returns an image Part.
// MIME is derived from the file extension; unknown extensions yield
// an empty MIME and the bytes are still wrapped (the C side will
// surface its own error if the format is unsupported).
func ImageFromFile(path string) (Part, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Part{}, fmt.Errorf("litertlm: ImageFromFile: %w", err)
	}
	return ImageWithMime(b, mimeForExt(filepath.Ext(path), imageMimeByExt)), nil
}

// Audio returns an audio Part wrapping b. The MIME type is left
// empty; pass AudioWithMime or AudioFromFile to set it.
func Audio(b []byte) Part {
	return Part{kind: partAudio, data: b}
}

// AudioWithMime returns an audio Part wrapping b with an explicit
// MIME type (e.g. "audio/wav", "audio/mpeg", "audio/ogg").
func AudioWithMime(b []byte, mime string) Part {
	return Part{kind: partAudio, data: b, mime: mime}
}

// AudioFromFile reads the file at path and returns an audio Part.
// MIME is derived from the file extension.
func AudioFromFile(path string) (Part, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Part{}, fmt.Errorf("litertlm: AudioFromFile: %w", err)
	}
	return AudioWithMime(b, mimeForExt(filepath.Ext(path), audioMimeByExt)), nil
}

// Mime returns the MIME type recorded on the Part. Empty for text
// parts and for binary parts constructed without a MIME.
func (p Part) Mime() string { return p.mime }

// IsText reports whether the Part holds text.
func (p Part) IsText() bool { return p.kind == partText }

// IsImage reports whether the Part holds image bytes.
func (p Part) IsImage() bool { return p.kind == partImage }

// IsAudio reports whether the Part holds audio bytes.
func (p Part) IsAudio() bool { return p.kind == partAudio }

// imageMimeByExt covers the formats LiteRT-LM's vision pipeline is
// known to accept. Unknown extensions return "" — the C side will
// fail on the actual call if the format truly isn't supported.
var imageMimeByExt = map[string]string{
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".png":  "image/png",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
}

var audioMimeByExt = map[string]string{
	".wav":  "audio/wav",
	".mp3":  "audio/mpeg",
	".ogg":  "audio/ogg",
	".flac": "audio/flac",
	".m4a":  "audio/mp4",
	".aac":  "audio/aac",
}

func mimeForExt(ext string, table map[string]string) string {
	return table[strings.ToLower(ext)]
}

// partsToInputs converts a high-level []Part to the low-level
// []InputData expected by Session.GenerateContent and
// Session.GenerateContentStreamCh. Image and audio parts emit their
// terminating End-marker automatically.
func partsToInputs(parts []Part) []InputData {
	out := make([]InputData, 0, len(parts)*2)
	for _, p := range parts {
		switch p.kind {
		case partText:
			out = append(out, NewTextInputString(p.text))
		case partImage:
			out = append(out,
				NewBinaryInput(InputImage, p.data),
				NewBinaryInput(InputImageEnd, nil),
			)
		case partAudio:
			out = append(out,
				NewBinaryInput(InputAudio, p.data),
				NewBinaryInput(InputAudioEnd, nil),
			)
		}
	}
	return out
}
