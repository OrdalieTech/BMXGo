package text_preprocessor

import (
	"reflect"
	"testing"
)

func TestWordTokenizer(t *testing.T) {
	text := "Hello, world!"
	expected := []string{"Hello", "world"}
	result := wordTokenizer(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("wordTokenizer(%q) = %v; want %v", text, result, expected)
	}
}

func TestWordPunctTokenizer(t *testing.T) {
	text := "Hello, world!"
	expected := []string{"Hello", ",", "world", "!"}
	result := wordPunctTokenizer(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("wordPunctTokenizer(%q) = %v; want %v", text, result, expected)
	}
}

func TestSentenceTokenizer(t *testing.T) {
	text := "Hello, world! How are you?"
	expected := []string{"Hello, world!", " How are you?"}
	result := sentenceTokenizer(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("sentenceTokenizer(%q) = %v; want %v", text, result, expected)
	}
}

func TestGetTokenizer(t *testing.T) {
	tests := []struct {
		name      string
		input     interface{}
		text      string
		expected  []string
		expectErr bool
	}{
		{"whitespace", "whitespace", "Hello world", []string{"Hello", "world"}, false},
		{"word", "word", "Hello, world!", []string{"Hello", "world"}, false},
		{"wordpunct", "wordpunct", "Hello, world!", []string{"Hello", ",", "world", "!"}, false},
		{"sent", "sent", "Hello, world! How are you?", []string{"Hello, world!", " How are you?"}, false},
		{"identity", nil, "Hello, world!", []string{"Hello, world!"}, false},
		{"unsupported", "unsupported", "Hello, world!", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer, err := GetTokenizer(tt.input)
			if (err != nil) != tt.expectErr {
				t.Fatalf("GetTokenizer(%v) error = %v, expectErr %v", tt.input, err, tt.expectErr)
			}
			if err == nil {
				result := tokenizer(tt.text)
				if !reflect.DeepEqual(result, tt.expected) {
					t.Errorf("tokenizer(%q) = %v; want %v", tt.text, result, tt.expected)
				}
			}
		})
	}
}
