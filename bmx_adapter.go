package BMXGo

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
)

func GenerateAugmentedQueries(query string, num_augmented_queries int) ([]string, error) {
	prompt := fmt.Sprintf(`You are an intelligent query augmentation tool. Your task is to augment each
query with {%d} similar queries and output JSONL format, like {”query”:
”original query”, ”augmented queries”: [”augmented query 1”, ”augmented
query 2”, ...]}
Input query: {%s}
Output:`, num_augmented_queries, query)

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
		// log.Println("AUGMENTED QUERIES GENERATED")
		// Clean up the response
		cleanResponse := strings.TrimSpace(responseTxt)
		cleanResponse = strings.TrimPrefix(cleanResponse, "```jsonl")
		cleanResponse = strings.TrimSuffix(cleanResponse, "```")

		// Parse the JSON response
		var result struct {
			Query            string   `json:"query"`
			AugmentedQueries []string `json:"augmented queries"`
		}

		err := json.Unmarshal([]byte(cleanResponse), &result)
		if err != nil {
			// If JSON parsing fails, try to extract queries using a simple string split
			queries := strings.Split(cleanResponse, "\",")
			if len(queries) > 1 {
				augmentedQueries := make([]string, 0, num_augmented_queries)
				for i, q := range queries[1:] { // Skip the first element (original query)
					if i >= num_augmented_queries {
						break
					}
					q = strings.Trim(q, "[] \"\n")
					if q != "" {
						augmentedQueries = append(augmentedQueries, q)
					}
				}
				return augmentedQueries, nil
			}
			return nil, fmt.Errorf("error parsing LLM response: %v", err)
		}

		// Combine original query with augmented queries
		augmentedQueries := append([]string{}, result.AugmentedQueries...)
		return augmentedQueries, nil
	case err := <-errChan:
		log.Printf("Error in generate_augmented_queries: %v", err)
		// Fallback: return the original query and some simple variations
		return GenerateAugmentedQueries(query, num_augmented_queries)
	}
}

type BMXAdapter struct {
	indexName string
	bmx       *BMX
}

type SearchResults struct {
	Keys   []string
	Scores []float64
}

func Build(indexName string, config Config) BMXAdapter {
	bmx := BMX{
		Docs:   map[string]Document{},
		Params: Parameters{},
	}
	bmx.InitializeTextPreprocessor(&config)

	return BMXAdapter{
		indexName: indexName,
		bmx:       &bmx,
	}
}

func (adapter *BMXAdapter) AddMany(ids []string, docs []string) error {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Recovered in AddMany", r)
			fmt.Println("Problematic document id:", ids[len(ids)-1])
		}
	}()

	tokenize := adapter.bmx.TextPreprocessor.Process

	for i, doc := range docs {
		adapter.bmx.Docs[ids[i]] = Document{Text: doc, Tokens: tokenize(doc)}
	}

	// fmt.Println("Setting params")
	// start := time.Now()
	adapter.bmx.SetParams()
	// fmt.Println("Parameters set, total time:", time.Since(start))
	// start = time.Now()
	// fmt.Println("Filling F table")
	adapter.bmx.F_table_fill()
	// fmt.Println("F table filled, total time:", time.Since(start))
	// start = time.Now()
	// fmt.Println("Calculating number of appearances")
	adapter.bmx.NumAppearancesCalc()
	// fmt.Println("Number of appearances calculated, total time:", time.Since(start))
	// start = time.Now()
	// fmt.Println("Filling IDF table")
	adapter.bmx.IDF_table_fill()
	// fmt.Println("IDF table filled, total time:", time.Since(start))
	// start = time.Now()
	// fmt.Println("Filling E_tilde table")
	adapter.bmx.E_tilde_table_fill()
	// fmt.Println("E_tilde table filled, total time:", time.Since(start))
	return nil
}

func (adapter *BMXAdapter) Search(query string, topK int) SearchResults {
	q := Query{Text: query}
	q.Initialize(adapter.bmx)

	Keys := make([]string, 0, len(q.ScoreTable))
	for key := range q.ScoreTable {
		Keys = append(Keys, key)
	}

	sort.Slice(Keys, func(i, j int) bool {
		return q.ScoreTable[Keys[i]] > q.ScoreTable[Keys[j]]
	})

	topKeys := Keys[:topK]

	var wg sync.WaitGroup
	topScores := make([]float64, len(topKeys))

	for i, key := range topKeys {
		wg.Add(1)
		go func(i int, key string) {
			defer wg.Done()
			topScores[i] = q.NormalizedScoreTable[key]
		}(i, key)
	}
	wg.Wait()

	return SearchResults{Keys: topKeys, Scores: topScores}
}

func (adapter *BMXAdapter) SearchMany(queries []string, topK int, maxConcurrent int) []SearchResults {
	results := make([]SearchResults, len(queries))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, maxConcurrent)

	for i, query := range queries {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore
		go func(i int, query string) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore
			results[i] = adapter.Search(query, topK)
			// if i%100 == 0 {
			// 	fmt.Printf("Query number %d out of %d\n", i+1, len(queries))
			// }
		}(i, query)
	}

	wg.Wait()
	close(semaphore)
	return results
}

func (adapter *BMXAdapter) SearchAugmented(query string, topK int, num_augmented_queries int, weight float64) SearchResults {
	// fmt.Println("Generating augmented queries")
	// start := time.Now()
	augmentedQueries, err := GenerateAugmentedQueries(query, num_augmented_queries)
	if err != nil {
		log.Fatal(err)
	}
	// fmt.Println("Augmented queries generated, total time:", time.Since(start))
	q := Query{Text: query, AugmentedQueries: augmentedQueries}

	q.AugmentedWeights = []float64{}
	for range augmentedQueries {
		q.AugmentedWeights = append(q.AugmentedWeights, weight)
	}

	// fmt.Println("Initializing query")
	// start = time.Now()
	q.Initialize(adapter.bmx)
	// fmt.Println("Query initialized, total time:", time.Since(start))

	// fmt.Println("Sorting keys")
	// start = time.Now()
	Keys := []string{}
	for key := range q.ScoreTable {
		Keys = append(Keys, key)
	}

	sort.Slice(Keys, func(i, j int) bool {
		return q.ScoreTable[Keys[i]] > q.ScoreTable[Keys[j]]
	})

	// fmt.Println("Keys sorted, total time:", time.Since(start))

	// fmt.Println("Selecting top keys")
	// start = time.Now()
	topKeys := Keys[:topK]

	var wg sync.WaitGroup
	topScores := make([]float64, len(topKeys))

	for i, key := range topKeys {
		wg.Add(1)
		go func(i int, key string) {
			defer wg.Done()
			topScores[i] = q.NormalizedScoreTable[key]
		}(i, key)
	}
	wg.Wait()

	return SearchResults{Keys: topKeys, Scores: topScores}
}

func (adapter *BMXAdapter) SearchAugmentedMany(queries []string, topK int, num_augmented_queries int, weight float64, maxConcurrent int) []SearchResults {
	results := make([]SearchResults, len(queries))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, maxConcurrent)

	for i, query := range queries {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore
		go func(i int, query string) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore
			results[i] = adapter.SearchAugmented(query, topK, num_augmented_queries, weight)
			// if i%100 == 99 {
			// 	fmt.Printf("Query number %d out of %d\n", i+1, len(queries))
			// }
		}(i, query)
	}

	wg.Wait()
	close(semaphore)
	return results
}

func (adapter *BMXAdapter) GetTokens(text string) {
	list := adapter.bmx.TextPreprocessor.Process(text)
	fmt.Println(list)
}
