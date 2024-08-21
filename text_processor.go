package BMXGo

import (
	"strings"
	"sync"
	"unicode"
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

// NewConfig creates a new Config with default values.
func NewConfig() *Config {
	return &Config{
		Tokenizer:                   strings.Fields,
		Stopwords:                   make(map[string]struct{}),
		DoLowercasing:               true,
		DoAmpersandNormalization:    true,
		DoSpecialCharsNormalization: true,
		DoAcronymsNormalization:     true,
		DoPunctuationRemoval:        true,
	}
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
		tp.steps = append(tp.steps, strings.ToLower)
	}
	if tp.config.DoAmpersandNormalization {
		tp.steps = append(tp.steps, normalizeAmpersand)
	}
	if tp.config.DoSpecialCharsNormalization {
		tp.steps = append(tp.steps, normalizeSpecialChars)
	}
	if tp.config.DoAcronymsNormalization {
		tp.steps = append(tp.steps, normalizeAcronyms)
	}
	if tp.config.DoPunctuationRemoval {
		tp.steps = append(tp.steps, removePunctuation)
	}
	// Remove tokenizer from tp.steps
	if len(tp.config.Stopwords) > 0 {
		tp.steps = append(tp.steps, tp.removeStopwords)
	}
	if tp.config.Stemmer != nil {
		tp.steps = append(tp.steps, tp.applyStemmer)
	}
	tp.steps = append(tp.steps, removeEmpty)
}

// Process processes a single text item through all preprocessing steps.
func (tp *TextPreprocessor) Process(item string) []string {
	for _, step := range tp.steps {
		item = step(item)
	}
	// Apply tokenizer separately
	return tp.config.Tokenizer(item)
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

// Helper functions for preprocessing steps
func normalizeAmpersand(s string) string {
	return strings.ReplaceAll(s, "&", "and")
}

func normalizeSpecialChars(s string) string {
	// Add your normalization logic here
	return s
}

func normalizeAcronyms(s string) string {
	// Add your normalization logic here
	return s
}

func removePunctuation(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
}

func removeEmpty(s string) string {
	return strings.TrimSpace(s)
}

func (tp *TextPreprocessor) removeStopwords(s string) string {
	words := strings.Fields(s)
	var filtered []string
	for _, word := range words {
		if _, found := tp.config.Stopwords[word]; !found {
			filtered = append(filtered, word)
		}
	}
	return strings.Join(filtered, " ")
}

func (tp *TextPreprocessor) applyStemmer(s string) string {
	words := strings.Fields(s)
	for i, word := range words {
		words[i] = tp.config.Stemmer(word)
	}
	return strings.Join(words, " ")
}
