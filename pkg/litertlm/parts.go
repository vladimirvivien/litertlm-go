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
// Construct via Text, Image, ImageWithMIME, ImageFromFile, Audio,
// AudioWithMIME, or AudioFromFile.
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

// Image returns an image Part wrapping b with no MIME metadata.
// Caller must treat b as read-only after constructing the Part.
func Image(b []byte) Part {
	return Part{kind: partImage, data: b}
}

// ImageWithMIME returns an image Part wrapping b with an explicit
// MIME type (e.g. "image/jpeg", "image/png", "image/webp").
// Caller must treat b as read-only after constructing the Part.
func ImageWithMIME(b []byte, mime string) Part {
	return Part{kind: partImage, data: b, mime: mime}
}

// ImageFromFile reads the file at path and returns an image Part.
// MIME is derived from the file extension.
func ImageFromFile(path string) (Part, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Part{}, fmt.Errorf("litertlm: ImageFromFile: %w", err)
	}
	return ImageWithMIME(b, mimeForExt(filepath.Ext(path), imageMimeByExt)), nil
}

// Audio returns an audio Part wrapping b with no MIME metadata.
// Caller must treat b as read-only after constructing the Part.
func Audio(b []byte) Part {
	return Part{kind: partAudio, data: b}
}

// AudioWithMIME returns an audio Part wrapping b with an explicit
// MIME type (e.g. "audio/wav", "audio/mpeg", "audio/ogg").
// Caller must treat b as read-only after constructing the Part.
func AudioWithMIME(b []byte, mime string) Part {
	return Part{kind: partAudio, data: b, mime: mime}
}

// AudioFromFile reads the file at path and returns an audio Part.
// MIME is derived from the file extension.
func AudioFromFile(path string) (Part, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Part{}, fmt.Errorf("litertlm: AudioFromFile: %w", err)
	}
	return AudioWithMIME(b, mimeForExt(filepath.Ext(path), audioMimeByExt)), nil
}

// MIME returns the MIME type recorded on the Part. Empty for text
// parts and for binary parts constructed without a MIME type.
func (p Part) MIME() string { return p.mime }

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

// assistantText extracts and concatenates all text-typed content items from the
// assistant reply envelope returned by Conversation.SendMessage.
func assistantText(envelope string) (string, error) {
	var env struct {
		Content []replyContentPart `json:"content"`
	}
	if err := json.Unmarshal([]byte(envelope), &env); err != nil {
		return "", fmt.Errorf("litertlm: parse assistant envelope: %w", err)
	}
	text, ok := concatTextParts(env.Content)
	if !ok {
		return "", fmt.Errorf("litertlm: no text part in assistant reply: %s", envelope)
	}
	return text, nil
}
