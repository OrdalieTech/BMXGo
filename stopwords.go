package BMXGo

import (
	"errors"
	"strings"
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
		return nil, errors.New("stop-words for " + strings.Title(lang) + " are not available")
	}
	// Placeholder for actual stopwords fetching logic
	// In a real implementation, you would fetch the stopwords from a file or database
	return []string{"example", "stopword"}, nil
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
