package litertlm

import (
	"reflect"
	"strings"
	"testing"
)

type schemaFlat struct {
	Name string
	Age  int
}

type schemaTagged struct {
	UserName string `json:"name"`
	UserID   int    `json:"id,omitempty"`
}

type schemaIgnored struct {
	Public string
	Hidden string `json:"-"`
}

type schemaUnexported struct {
	Public  string
	private string //nolint:unused
}

type schemaNested struct {
	Title       string   `json:"title"`
	Ingredients []string `json:"ingredients"`
}

type schemaDeep struct {
	Items []schemaNested `json:"items"`
}

type schemaWithMap struct {
	Counts map[string]int `json:"counts"`
}

type schemaWithPointer struct {
	Friend *schemaFlat `json:"friend"`
}

type schemaWithBool struct {
	Active bool
	Score  float64
}

func TestShapeOf_Table(t *testing.T) {
	tests := []struct {
		name string
		typ  reflect.Type
		want string
	}{
		{"flat", reflect.TypeOf(schemaFlat{}),
			`{"name": <string>, "age": <number>}`},
		{"tagged", reflect.TypeOf(schemaTagged{}),
			`{"name": <string>, "id": <number>}`},
		{"ignored field", reflect.TypeOf(schemaIgnored{}),
			`{"public": <string>}`},
		{"unexported field", reflect.TypeOf(schemaUnexported{}),
			`{"public": <string>}`},
		{"nested", reflect.TypeOf(schemaNested{}),
			`{"title": <string>, "ingredients": [<string>]}`},
		{"deep", reflect.TypeOf(schemaDeep{}),
			`{"items": [{"title": <string>, "ingredients": [<string>]}]}`},
		{"map", reflect.TypeOf(schemaWithMap{}),
			`{"counts": {"<key>": <number>}}`},
		{"pointer field", reflect.TypeOf(schemaWithPointer{}),
			`{"friend": {"name": <string>, "age": <number>}}`},
		{"bool and float", reflect.TypeOf(schemaWithBool{}),
			`{"active": <boolean>, "score": <number>}`},
		{"top-level slice", reflect.TypeOf([]schemaFlat{}),
			`[{"name": <string>, "age": <number>}]`},
		{"top-level array", reflect.TypeOf([3]int{}),
			`[<number>]`},
		{"top-level pointer", reflect.TypeOf((*schemaFlat)(nil)),
			`{"name": <string>, "age": <number>}`},
		{"plain string", reflect.TypeOf(""),
			`<string>`},
		{"plain int", reflect.TypeOf(0),
			`<number>`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := shapeOf(tt.typ)
			if err != nil {
				t.Fatalf("shapeOf: unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got  = %s\nwant = %s", got, tt.want)
			}
		})
	}
}

func TestShapeOf_UnsupportedKinds(t *testing.T) {
	type withChan struct {
		Ch chan int
	}
	type withFunc struct {
		Fn func()
	}
	type withMapNonStringKey struct {
		M map[int]string
	}

	for _, tt := range []struct {
		name string
		typ  reflect.Type
	}{
		{"channel", reflect.TypeOf(withChan{})},
		{"func", reflect.TypeOf(withFunc{})},
		{"map[int]X", reflect.TypeOf(withMapNonStringKey{})},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := shapeOf(tt.typ)
			if err == nil {
				t.Errorf("expected error for unsupported type")
			}
		})
	}
}

// TestShapeOf_RecursionLimit guards against pathological self-referential
// types triggering unbounded recursion. The recursion limit short-circuits
// well before stack overflow.
func TestShapeOf_RecursionLimit(t *testing.T) {
	type rec struct {
		Self *rec
	}
	// `*rec` unwraps to `rec`, which has `*rec` again. Each level
	// increments depth; we should hit the cap and return an error
	// rather than blowing the stack.
	_, err := shapeOf(reflect.TypeOf(rec{}))
	if err == nil || !strings.Contains(err.Error(), "deeper than") {
		t.Errorf("expected recursion-limit error, got %v", err)
	}
}

func TestIsArrayType(t *testing.T) {
	cases := []struct {
		typ  reflect.Type
		want bool
	}{
		{reflect.TypeOf(""), false},
		{reflect.TypeOf(0), false},
		{reflect.TypeOf(schemaFlat{}), false},
		{reflect.TypeOf([]int{}), true},
		{reflect.TypeOf([3]int{}), true},
		{reflect.TypeOf((*[]int)(nil)), true},
		{reflect.TypeOf((*schemaFlat)(nil)), false},
	}
	for _, c := range cases {
		if got := isArrayType(c.typ); got != c.want {
			t.Errorf("isArrayType(%v) = %v, want %v", c.typ, got, c.want)
		}
	}
}

func TestJSONFieldName(t *testing.T) {
	type s struct {
		A string
		B string `json:"renamed"`
		C string `json:"with_modifiers,omitempty,string"`
		D string `json:"-"`
		E string `json:",omitempty"`
	}
	st := reflect.TypeOf(s{})
	cases := []struct {
		field string
		want  string
	}{
		{"A", "a"},
		{"B", "renamed"},
		{"C", "with_modifiers"},
		{"D", "-"},
		{"E", "e"},
	}
	for _, c := range cases {
		f, _ := st.FieldByName(c.field)
		if got := jsonFieldName(f); got != c.want {
			t.Errorf("jsonFieldName(%s) = %q, want %q", c.field, got, c.want)
		}
	}
}
