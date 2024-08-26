package BMXGo

import (
	"fmt"
	"strings"
	"sync"
)

// Config holds the configuration for text preprocessing.
type Config struct {
	Tokenizer                   func(string) []string
	Stemmer                     func(string) string
	Stopwords                   map[string]struct{}
	DoLowercasing               bool
	DoAmpersandNormalization    bool
	DoSpecialCharsNormalization bool
	DoAcronymsNormalization     bool
	DoPunctuationRemoval        bool
}

// NewConfig creates a new Config with the specified tokenizer, stemmer, and stopwords.
func NewConfig(tokenizer string, stemmer string, lang string) (*Config, error) {
	tokenizerFunc, err := GetTokenizer(tokenizer)
	if err != nil {
		return nil, fmt.Errorf("error getting tokenizer: %w", err)
	}

	stemmerFunc, err := GetStemmer(stemmer)
	if err != nil {
		return nil, fmt.Errorf("error getting stemmer: %w", err)
	}

	config := &Config{
		Tokenizer:                   tokenizerFunc,
		Stemmer:                     stemmerFunc,
		Stopwords:                   make(map[string]struct{}),
		DoLowercasing:               true,
		DoAmpersandNormalization:    true,
		DoSpecialCharsNormalization: true,
		DoAcronymsNormalization:     false,
		DoPunctuationRemoval:        true,
	}

	stopwords, err := GetStopwords(lang)
	if err != nil {
		return nil, fmt.Errorf("error getting stopwords: %w", err)
	}

	for _, word := range stopwords {
		config.Stopwords[word] = struct{}{}
	}

	return config, nil
}

// TextPreprocessor holds the preprocessing steps and configuration.
type TextPreprocessor struct {
	config *Config
	steps  []func(string) string
}

// NewTextPreprocessor creates a new TextPreprocessor with the given configuration.
func NewTextPreprocessor(config *Config) *TextPreprocessor {
	tp := &TextPreprocessor{config: config}
	tp.createPreprocessingSteps()
	return tp
}

// createPreprocessingSteps creates the preprocessing steps based on the configuration.
func (tp *TextPreprocessor) createPreprocessingSteps() {
	if tp.config.DoLowercasing {
		tp.steps = append(tp.steps, Lowercasing)
	}
	if tp.config.DoAmpersandNormalization {
		tp.steps = append(tp.steps, NormalizeAmpersand)
	}
	if tp.config.DoSpecialCharsNormalization {
		tp.steps = append(tp.steps, NormalizeSpecialChars)
	}
	if tp.config.DoAcronymsNormalization {
		tp.steps = append(tp.steps, NormalizeAcronyms)
	}
	if tp.config.DoPunctuationRemoval {
		tp.steps = append(tp.steps, RemovePunctuation)
	}
	tp.steps = append(tp.steps, NormalizeDiacritics)
	tp.steps = append(tp.steps, StripWhitespaces)
	// Remove tokenizer from tp.steps
	if len(tp.config.Stopwords) > 0 {
		tp.steps = append(tp.steps, tp.removeStopwords)
	}
	if tp.config.Stemmer != nil {
		tp.steps = append(tp.steps, tp.applyStemmer)
	}
}

// Process processes a single text item through all preprocessing steps.
func (tp *TextPreprocessor) Process(item string) []string {
	for _, step := range tp.steps {
		item = step(item)
	}
	// Apply tokenizer separately
	tokens := tp.config.Tokenizer(item)
	finalTokens := RemoveEmptyTokens(tokens)
	return finalTokens
}

// ProcessMany processes multiple text items concurrently.
func (tp *TextPreprocessor) ProcessMany(items []string, nWorkers int) [][]string {
	var wg sync.WaitGroup
	out := make([][]string, len(items))
	ch := make(chan struct {
		index int
		item  string
	}, len(items))

	for i, item := range items {
		ch <- struct {
			index int
			item  string
		}{index: i, item: item}
	}
	close(ch)

	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range ch {
				out[job.index] = tp.Process(job.item)
			}
		}()
	}

	wg.Wait()
	return out
}

// Helper functions for preprocessing steps can be removed as they are now in normalization.go

// Update removeStopwords to use the RemoveStopwords function from stopwords.go
func (tp *TextPreprocessor) removeStopwords(s string) string {
	tokens := strings.Fields(s)
	filteredTokens := RemoveStopwords(tokens, tp.config.Stopwords)
	return strings.Join(filteredTokens, " ")
}

// Update applyStemmer to use the ApplyStemmer function from stemmer.go
func (tp *TextPreprocessor) applyStemmer(s string) string {
	tokens := strings.Fields(s)
	stemmedTokens := ApplyStemmer(tokens, tp.config.Stemmer)
	return strings.Join(stemmedTokens, " ")
}

// Add a method to set the stemmer
func (tp *TextPreprocessor) SetStemmer(stemmerName string) error {
	stemmer, err := GetStemmer(stemmerName)
	if err != nil {
		return err
	}
	tp.config.Stemmer = stemmer
	tp.createPreprocessingSteps() // Recreate steps to include the stemmer
	return nil
}

// Add a method to set stopwords
func (tp *TextPreprocessor) SetStopwords(stopwords interface{}) error {
	stopwordsList, err := GetStopwords(stopwords)
	if err != nil {
		return err
	}
	tp.config.Stopwords = make(map[string]struct{})
	for _, word := range stopwordsList {
		tp.config.Stopwords[word] = struct{}{}
	}
	tp.createPreprocessingSteps() // Recreate steps to include stopwords removal
	return nil
}
