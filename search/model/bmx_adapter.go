package model

import (
	"BMXGo/search/text_preprocessor"
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
)

func GenerateAugmentedQueries(query string, num_augmented_queries int) ([]string, error) {
	prompt := fmt.Sprintf(`Étant donné la requête suivante, générez une liste de %d requêtes augmentées qui incluent des synonymes et des termes pertinents pour améliorer une recherche lexicale. Chaque requête augmentée doit être sur une nouvelle ligne. N'inclus pas la requête originale dans la liste des requêtes augmentées.

Exemple :
Requête originale : "congé de maternité"

Requêtes augmentées :
congés parentaux
congés de grossesse
congés pour naissance


Requête originale : %s

Requêtes augmentées :`, num_augmented_queries, query)

	client := NewLLMClient(ClientConfig{
		Provider:       "openai",
		DeploymentName: "gpt-4o-mini",
	})
	request := ChatCompletionRequest{
		Models: []string{"gpt-4o-mini"},
		Messages: []ConvMessage{
			{Role: "user", Content: strings.TrimSpace(prompt)},
		},
		Stream:      false,
		Temperature: 0.7,
		MaxTokens:   200,
	}

	respChan, errChan := client.Completion(context.Background(), request)

	select {
	case responseTxt := <-respChan:
		log.Println("AUGMENTED QUERIES GENERATED")
		augmentedQueries := strings.Split(strings.TrimSpace(responseTxt), "\n")
		augmentedQueries = append([]string{query}, augmentedQueries...)
		return augmentedQueries, nil
	case err := <-errChan:
		return nil, fmt.Errorf("error in generate_augmented_queries: %v", err)
	}
}

type BMXAdapter struct {
	indexName string
	bmx       *BMX
}

type SearchResults struct {
	keys   []string
	scores []float64
}

func Build(indexName string) BMXAdapter {
	return BMXAdapter{indexName: indexName, bmx: &BMX{}}
}

func (adapter *BMXAdapter) AddMany(ids []string, docs []string) {
	tokenize, err := text_preprocessor.GetTokenizer("word")
	if err != nil {
		log.Fatal(err)
	}
	for i, doc := range docs {
		adapter.bmx.Docs[ids[i]] = Document{Text: doc, Tokens: tokenize(doc)}
	}
}

func (adapter *BMXAdapter) Search(query string, topK int) SearchResults {
	tokenize, err := text_preprocessor.GetTokenizer("word")
	if err != nil {
		log.Fatal(err)
	}
	adapter.bmx.Query = Query{Text: query, Tokens: tokenize(query)}
	adapter.bmx.AugmentedQueries = []Query{}
	adapter.bmx.AugmentedWeights = []float64{}
	adapter.bmx.Initialize()

	Keys := []string{}
	for key := range adapter.bmx.Query.Score_table {
		Keys = append(Keys, key)
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(Keys, func(i, j int) bool {
		return adapter.bmx.Query.Score_table[Keys[i]] > adapter.bmx.Query.Score_table[Keys[j]]
	})

	topKeys := Keys[:topK]

	topScores := []float64{}
	for _, key := range topKeys {
		topScores = append(topScores, adapter.bmx.Query.NormalizedScore_table[key])
	}
	return SearchResults{keys: topKeys, scores: topScores}
}

func (adapter *BMXAdapter) SearchMany(queries []string, topK int) []SearchResults {
	results := []SearchResults{}
	for _, query := range queries {
		results = append(results, adapter.Search(query, topK))
	}
	return results
}

func (adapter *BMXAdapter) SearchAugmented(query string, topK int, num_augmented_queries int) SearchResults {
	tokenize, err := text_preprocessor.GetTokenizer("word")
	if err != nil {
		log.Fatal(err)
	}
	adapter.bmx.Query = Query{Text: query, Tokens: tokenize(query)}

	augmentedQueries, err := GenerateAugmentedQueries(query, num_augmented_queries)
	if err != nil {
		log.Fatal(err)
	}

	for _, augmentedQuery := range augmentedQueries {
		if query != augmentedQuery {
			q := Query{Text: augmentedQuery}
			q.Tokens = tokenize(q.Text)
			adapter.bmx.AugmentedQueries = append(adapter.bmx.AugmentedQueries, q)
		}
	}

	adapter.bmx.AugmentedWeights = []float64{}
	for range augmentedQueries {
		adapter.bmx.AugmentedWeights = append(adapter.bmx.AugmentedWeights, 1.0/float64(len(augmentedQueries)+1))
	}

	adapter.bmx.Initialize()

	Keys := []string{}
	for key := range adapter.bmx.AugmentedScoreTable {
		Keys = append(Keys, key)
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(Keys, func(i, j int) bool {
		return adapter.bmx.AugmentedScoreTable[Keys[i]] > adapter.bmx.AugmentedScoreTable[Keys[j]]
	})

	topKeys := Keys[:topK]

	topScores := []float64{}
	for _, key := range topKeys {
		topScores = append(topScores, adapter.bmx.NormalizedScoreTable[key])
	}
	return SearchResults{keys: topKeys, scores: topScores}
}

func (adapter *BMXAdapter) SearchAugmentedMany(queries []string, topK int, num_augmented_queries int) []SearchResults {
	results := []SearchResults{}
	for _, query := range queries {
		results = append(results, adapter.SearchAugmented(query, topK, num_augmented_queries))
	}
	return results
}
