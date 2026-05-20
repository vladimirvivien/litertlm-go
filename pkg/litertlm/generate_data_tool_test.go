package litertlm

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

// ---- types used across tests --------------------------------------------

type capturePerson struct {
	Name string `description:"the person's full name"`
	Age  int    `description:"age in years"`
}

type captureAddress struct {
	Street string
	City   string
}

// ---- captureToolName ----------------------------------------------------

func TestCaptureToolName_FormatAndPrefix(t *testing.T) {
	name := captureToolName[capturePerson]()
	if !strings.HasPrefix(name, reservedToolNamePrefix) {
		t.Errorf("captureToolName = %q; want prefix %q", name, reservedToolNamePrefix)
	}
	if !strings.HasPrefix(name, reservedToolNamePrefix+"capture_") {
		t.Errorf("captureToolName = %q; want %qcapture_<hash>", name, reservedToolNamePrefix)
	}
	if got := captureToolName[capturePerson](); got != name {
		t.Errorf("captureToolName not stable: %q vs %q", got, name)
	}
}

func TestCaptureToolName_DistinctPerType(t *testing.T) {
	a := captureToolName[capturePerson]()
	b := captureToolName[captureAddress]()
	if a == b {
		t.Errorf("distinct types produced the same capture name: %q", a)
	}
}

// ---- getOrSynthesizeCaptureTool -----------------------------------------

func TestGetOrSynthesizeCaptureTool_CachesByType(t *testing.T) {
	c := &Client{}
	first, err := getOrSynthesizeCaptureTool[capturePerson](c)
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	second, err := getOrSynthesizeCaptureTool[capturePerson](c)
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if first != second {
		t.Errorf("getOrSynthesize did not cache: %p vs %p", first, second)
	}
	if len(c.tools) != 1 {
		t.Errorf("expected one tool registered, got %d: %v", len(c.tools), c.tools)
	}
}

func TestGetOrSynthesizeCaptureTool_DistinctTypes(t *testing.T) {
	c := &Client{}
	p, err := getOrSynthesizeCaptureTool[capturePerson](c)
	if err != nil {
		t.Fatalf("person: %v", err)
	}
	a, err := getOrSynthesizeCaptureTool[captureAddress](c)
	if err != nil {
		t.Fatalf("address: %v", err)
	}
	if p.Name() == a.Name() {
		t.Errorf("distinct types share tool name %q", p.Name())
	}
	if len(c.tools) != 2 {
		t.Errorf("expected two tools, got %d", len(c.tools))
	}
}

func TestGetOrSynthesizeCaptureTool_PointerStructAccepted(t *testing.T) {
	c := &Client{}
	tool, err := getOrSynthesizeCaptureTool[*capturePerson](c)
	if err != nil {
		t.Fatalf("pointer-to-struct should be accepted: %v", err)
	}
	if tool.Parameters()["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", tool.Parameters()["type"])
	}
}

func TestGetOrSynthesizeCaptureTool_RejectsNonStruct(t *testing.T) {
	c := &Client{}
	cases := []func(*Client) error{
		func(c *Client) error { _, err := getOrSynthesizeCaptureTool[string](c); return err },
		func(c *Client) error { _, err := getOrSynthesizeCaptureTool[int](c); return err },
		func(c *Client) error { _, err := getOrSynthesizeCaptureTool[[]capturePerson](c); return err },
		func(c *Client) error {
			_, err := getOrSynthesizeCaptureTool[map[string]capturePerson](c)
			return err
		},
	}
	for i, fn := range cases {
		err := fn(c)
		if !errors.Is(err, errCaptureToolUnsuitable) {
			t.Errorf("case %d: err = %v; want errCaptureToolUnsuitable", i, err)
		}
	}
	if len(c.tools) != 0 {
		t.Errorf("non-struct cases registered tools: %v", c.tools)
	}
}

func TestGetOrSynthesizeCaptureTool_NilClient(t *testing.T) {
	_, err := getOrSynthesizeCaptureTool[capturePerson](nil)
	if err == nil {
		t.Error("expected error for nil client")
	}
}

func TestGetOrSynthesizeCaptureTool_ConcurrentSameType(t *testing.T) {
	c := &Client{}
	const N = 32
	var wg sync.WaitGroup
	results := make([]*ManagedTool[capturePerson, struct{}], N)
	errs := make([]error, N)
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = getOrSynthesizeCaptureTool[capturePerson](c)
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: %v", i, err)
		}
	}
	first := results[0]
	for i, r := range results[1:] {
		if r != first {
			t.Errorf("goroutine %d returned %p, want %p", i+1, r, first)
		}
	}
	if len(c.tools) != 1 {
		t.Errorf("expected one tool after concurrent synthesis, got %d", len(c.tools))
	}
}

// ---- captureHandler -----------------------------------------------------

func TestCaptureHandler_WritesToCtxSlot(t *testing.T) {
	handler := captureHandler[capturePerson]()
	var slot *capturePerson
	ctx := context.WithValue(context.Background(), captureKey[capturePerson]{}, &slot)
	want := capturePerson{Name: "Alice", Age: 30}
	if _, err := handler(ctx, want); err != nil {
		t.Fatalf("handler: %v", err)
	}
	if slot == nil {
		t.Fatal("slot still nil after handler call")
	}
	if *slot != want {
		t.Errorf("captured = %+v, want %+v", *slot, want)
	}
}

func TestCaptureHandler_NoSlotInCtxIsNoop(t *testing.T) {
	handler := captureHandler[capturePerson]()
	if _, err := handler(context.Background(), capturePerson{Name: "Bob"}); err != nil {
		t.Errorf("handler with no slot returned error: %v", err)
	}
}

func TestCaptureHandler_WrongTypeSlotIgnored(t *testing.T) {
	handler := captureHandler[capturePerson]()
	var wrongSlot *captureAddress
	ctx := context.WithValue(context.Background(), captureKey[captureAddress]{}, &wrongSlot)
	if _, err := handler(ctx, capturePerson{Name: "Carol"}); err != nil {
		t.Errorf("handler should ignore foreign-type slots: %v", err)
	}
	if wrongSlot != nil {
		t.Errorf("foreign-type slot was written: %+v", wrongSlot)
	}
}

func TestCaptureHandler_ConcurrentDistinctSlots(t *testing.T) {
	handler := captureHandler[capturePerson]()
	const N = 16
	var wg sync.WaitGroup
	slots := make([]*capturePerson, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.WithValue(context.Background(), captureKey[capturePerson]{}, &slots[i])
			_, _ = handler(ctx, capturePerson{Name: "p", Age: i})
		}(i)
	}
	wg.Wait()
	for i, s := range slots {
		if s == nil {
			t.Errorf("slot %d not written", i)
			continue
		}
		if s.Age != i {
			t.Errorf("slot %d age = %d, want %d", i, s.Age, i)
		}
	}
}
