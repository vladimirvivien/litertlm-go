package litertlm

import (
	"fmt"
	"reflect"
	"strings"
)

// shapeOf returns a compact JSON-shape hint for type t. The output is
// not formal JSON Schema — it's an instruction-friendly hint format
// that local LLMs follow more reliably than a full schema document.
//
// Examples:
//
//	struct{Name string; Age int}      → {"name": <string>, "age": <number>}
//	[]string                          → [<string>]
//	*Recipe                           → (recursed)
//	map[string]int                    → {"<key>": <number>}
//
// Unsupported kinds (channel, interface, func, complex, unsafe.Pointer)
// return an error so callers see the failure at GenerateData time
// rather than from a confused model.
func shapeOf(t reflect.Type) (string, error) {
	return shapeOfDepth(t, 0)
}

func shapeOfDepth(t reflect.Type, depth int) (string, error) {
	if depth > 32 {
		return "", fmt.Errorf("schema: type %s nests deeper than 32 levels", t)
	}

	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return "<string>", nil
	case reflect.Bool:
		return "<boolean>", nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return "<number>", nil
	case reflect.Slice, reflect.Array:
		elem, err := shapeOfDepth(t.Elem(), depth+1)
		if err != nil {
			return "", err
		}
		return "[" + elem + "]", nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return "", fmt.Errorf("schema: map key must be string; got %s", t.Key())
		}
		val, err := shapeOfDepth(t.Elem(), depth+1)
		if err != nil {
			return "", err
		}
		return `{"<key>": ` + val + "}", nil
	case reflect.Struct:
		var b strings.Builder
		b.WriteString("{")
		first := true
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name := jsonFieldName(f)
			if name == "-" {
				continue
			}
			if !first {
				b.WriteString(", ")
			}
			first = false
			b.WriteString(`"`)
			b.WriteString(name)
			b.WriteString(`": `)
			inner, err := shapeOfDepth(f.Type, depth+1)
			if err != nil {
				return "", err
			}
			b.WriteString(inner)
		}
		b.WriteString("}")
		return b.String(), nil
	default:
		return "", fmt.Errorf("schema: unsupported kind %s for type %s", t.Kind(), t)
	}
}

// jsonFieldName extracts the JSON field name from a struct field's
// `json` tag, falling back to the lowercased Go field name. Strips
// `,omitempty` and similar modifiers. Returns "-" for ignored fields.
func jsonFieldName(f reflect.StructField) string {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "-"
	}
	if tag == "" {
		return strings.ToLower(f.Name)
	}
	if comma := strings.IndexByte(tag, ','); comma >= 0 {
		tag = tag[:comma]
	}
	if tag == "" {
		return strings.ToLower(f.Name)
	}
	return tag
}

// isArrayType reports whether t (after pointer unwrap) is a slice or
// array. Used to pick top-level [...] vs {...} for the shape hint and
// for the JSON extractor.
func isArrayType(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Kind() == reflect.Slice || t.Kind() == reflect.Array
}
