package BMXGo

import (
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/kljensen/snowball"
	"github.com/reiver/go-porterstemmer"
)

// Define a type for the stemmer function
type StemmerFunc func(string) string

// Define a map to hold the stemmers
var stemmersDict = map[string]StemmerFunc{
	"porter":  porterStemmer,
	"english": snowballStemmer("english"),
	"french":  snowballStemmer("french"),
	"german":  snowballStemmer("german"),
	"spanish": snowballStemmer("spanish"),
	"russian": snowballStemmer("russian"),
	"swedish": snowballStemmer("swedish"),
	"turkish": snowballStemmer("turkish"),
	// Add more stemmers as needed
}

// Porter stemmer implementation
func porterStemmer(word string) string {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Recovered in porterStemmer", r)
			fmt.Println("Problematic word:", word)
		}
	}()
	if word == "" {
		return ""
	}
	if len(word) < 6 {
		// Skip very short words or add them unchanged
		return word
	}
	stemmed := porterstemmer.StemString(word)
	if stemmed == "" {
		log.Printf("Warning: Porter stemmer returned empty string for input: %q", word)
		return word // Return the original word if stemming fails
	}
	return stemmed
}

// Snowball stemmer implementation
func snowballStemmer(language string) StemmerFunc {
	return func(word string) string {
		stemmed, err := snowball.Stem(word, language, true)
		if err != nil {
			return word
		}
		return stemmed
	}
}

// GetStemmer retrieves the appropriate stemmer function
func GetStemmer(stemmer string) (StemmerFunc, error) {
	stemmer = strings.ToLower(stemmer)
	if stemFunc, exists := stemmersDict[stemmer]; exists {
		return stemFunc, nil
	}
	return nil, errors.New("stemmer not supported")
}
