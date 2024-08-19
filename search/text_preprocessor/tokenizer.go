package text_preprocessor

import (
	"errors"
	"regexp"
	"strings"
)

// TokenizerFunc defines the type for tokenizer functions.
type TokenizerFunc func(string) []string

// tokenizersDict maps tokenizer names to their corresponding functions.
var tokenizersDict = map[string]TokenizerFunc{
	"whitespace": strings.Fields,
	"word":       wordTokenizer,
	"wordpunct":  wordPunctTokenizer,
	"sent":       sentenceTokenizer,
}

// wordTokenizer tokenizes text into words.
func wordTokenizer(text string) []string {
	re := regexp.MustCompile(`\w+`)
	return re.FindAllString(text, -1)
}

// wordPunctTokenizer tokenizes text into words and punctuation.
func wordPunctTokenizer(text string) []string {
	re := regexp.MustCompile(`\w+|[^\w\s]+`)
	return re.FindAllString(text, -1)
}

// sentenceTokenizer tokenizes text into sentences.
func sentenceTokenizer(text string) []string {
	re := regexp.MustCompile(`[^.!?]+[.!?]*`)
	return re.FindAllString(text, -1)
}

// getTokenizer returns the tokenizer function based on the provided name.
func getTokenizer(tokenizer string) (TokenizerFunc, error) {
	tokenizer = strings.ToLower(tokenizer)
	if fn, exists := tokenizersDict[tokenizer]; exists {
		return fn, nil
	}
	return nil, errors.New("tokenizer " + tokenizer + " not supported")
}

// GetTokenizer returns a tokenizer function based on the provided input.
func GetTokenizer(tokenizer interface{}) (TokenizerFunc, error) {
	switch t := tokenizer.(type) {
	case string:
		return getTokenizer(t)
	case TokenizerFunc:
		return t, nil
	case nil:
		return identityFunction, nil
	default:
		return nil, errors.New("not implemented")
	}
}

// identityFunction is a tokenizer function that returns the input as a single token.
func identityFunction(input string) []string {
	return []string{input}
}
