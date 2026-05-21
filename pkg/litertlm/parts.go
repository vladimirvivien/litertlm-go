package litertlm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Part is one piece of multimodal input to GenerateMulti /
// GenerateMultiStream / GenerateMultiResponse / GenerateDataMulti.
// Construct via Text, Image, ImageWithMime, ImageFromFile, Audio,
// AudioWithMime, or AudioFromFile.
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

// TextContent returns the text payload of a text Part. Empty string
// when the Part holds image or audio bytes; use IsText to disambiguate.
func (p Part) TextContent() string {
	if p.kind == partText {
		return p.text
	}
	return ""
}

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

// partsHasBinary reports whether parts contains an image or audio
// Part.
func partsHasBinary(parts []Part) bool {
	for _, p := range parts {
		if p.kind == partImage || p.kind == partAudio {
			return true
		}
	}
	return false
}

// renderPartsContent builds the content array the C side expects for
// a message: a list of typed entries, one per Part. Image and audio
// bytes are base64-encoded into a "blob" field; text becomes a "text"
// field. Used by both the current-turn multimodal path and the
// initial-history encoder.
func renderPartsContent(parts []Part) []map[string]any {
	content := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch p.kind {
		case partText:
			content = append(content, map[string]any{
				"type": "text",
				"text": p.text,
			})
		case partImage:
			content = append(content, map[string]any{
				"type": "image",
				"blob": base64.StdEncoding.EncodeToString(p.data),
			})
		case partAudio:
			content = append(content, map[string]any{
				"type": "audio",
				"blob": base64.StdEncoding.EncodeToString(p.data),
			})
		}
	}
	return content
}

// partsToConversationMessage builds the JSON user-message envelope
// the Conversation API expects for multimodal turns.
func partsToConversationMessage(parts []Part) (string, error) {
	msg := map[string]any{
		"role":    "user",
		"content": renderPartsContent(parts),
	}
	b, err := json.Marshal(msg)
	if err != nil {
		return "", fmt.Errorf("litertlm: marshal multimodal message: %w", err)
	}
	return string(b), nil
}

// assistantText extracts the first text-typed content item from the
// assistant reply envelope returned by Conversation.SendMessage:
// `{"role":"assistant","content":[{"type":"text","text":"..."}]}`.
func assistantText(envelope string) (string, error) {
	var env struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(envelope), &env); err != nil {
		return "", fmt.Errorf("litertlm: parse assistant envelope: %w", err)
	}
	for _, p := range env.Content {
		if p.Type == "text" {
			return p.Text, nil
		}
	}
	return "", fmt.Errorf("litertlm: no text part in assistant reply: %s", envelope)
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
