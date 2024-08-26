package text_preprocessor

import (
	"regexp"
	"strings"

	"github.com/rainycape/unidecode"
)

// Translation tables
var specialCharsTrans = strings.NewReplacer("‘", "'", "’", "'", "´", "'", "“", "\"", "”", "\"", "–", "-", "-", "-")
var punctuationTranslation = func() *strings.Replacer {
	var oldnew []string
	for _, r := range string([]rune{'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'}) {
		oldnew = append(oldnew, string(r), " ")
	}
	return strings.NewReplacer(oldnew...)
}()

func Lowercasing(text string) string {
	return strings.ToLower(text)
}

func NormalizeAmpersand(text string) string {
	return strings.ReplaceAll(text, "&", " and ")
}

func NormalizeDiacritics(text string) string {
	return unidecode.Unidecode(text)
}

func NormalizeSpecialChars(text string) string {
	return specialCharsTrans.Replace(text)
}

func NormalizeAcronyms(text string) string {
	re := regexp.MustCompile(`\.(?:[ \t\n\r\f\v]|$)`)
	return re.ReplaceAllString(text, "")
}

func RemovePunctuation(text string) string {
	return punctuationTranslation.Replace(text)
}

func StripWhitespaces(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

func RemoveEmptyTokens(tokens []string) []string {
	var result []string
	for _, token := range tokens {
		if token != "" {
			result = append(result, token)
		}
	}
	return result
}

func RemoveStopwords(tokens []string, stopwords map[string]struct{}) []string {
	var result []string
	for _, token := range tokens {
		if _, found := stopwords[token]; !found {
			result = append(result, token)
		}
	}
	return result
}

func ApplyStemmer(tokens []string, stemmer func(string) string) []string {
	var result []string
	for _, token := range tokens {
		result = append(result, stemmer(token))
	}
	return result
}
