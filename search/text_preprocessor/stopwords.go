package text_preprocessor

import (
	"errors"
	"strings"

	"bufio"
	"os"
	"path/filepath"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

var supportedLanguages = map[string]struct{}{
	"arabic":      {},
	"azerbaijani": {},
	"basque":      {},
	"bengali":     {},
	"catalan":     {},
	"chinese":     {},
	"danish":      {},
	"dutch":       {},
	"english":     {},
	"finnish":     {},
	"french":      {},
	"german":      {},
	"greek":       {},
	"hebrew":      {},
	"hinglish":    {},
	"hungarian":   {},
	"indonesian":  {},
	"italian":     {},
	"kazakh":      {},
	"nepali":      {},
	"norwegian":   {},
	"portuguese":  {},
	"romanian":    {},
	"russian":     {},
	"slovene":     {},
	"spanish":     {},
	"swedish":     {},
	"tajik":       {},
	"turkish":     {},
}

func getStopwords(lang string) ([]string, error) {
	lang = strings.ToLower(lang)
	if _, ok := supportedLanguages[lang]; !ok {
		return nil, errors.New("stop-words for " + cases.Title(language.Und).String(lang) + " are not available")
	}

	filename := filepath.Join("stopwords", lang+".txt")
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var stopwords []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		stopwords = append(stopwords, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return stopwords, nil
}

func GetStopwords(swList interface{}) ([]string, error) {
	switch v := swList.(type) {
	case string:
		return getStopwords(v)
	case []string:
		return v, nil
	case map[string]struct{}:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		return keys, nil
	case nil:
		return []string{}, nil
	default:
		return nil, errors.New("unsupported type for stopwords")
	}
}
