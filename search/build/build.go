package build

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"BMXGo/search/model"
	"BMXGo/search/text_preprocessor"
)

const SUMMARIZE_WITH_CONTEXT_PROMPT = "Your prompt template here"

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

// BuildBMX creates a BMX object from a query and a folder of text documents
func BuildBMX(query string, folderPath string, num_augmented_queries int) (*model.BMX, error) {
	// Initialize BMX struct
	bmx := &model.BMX{
		Query:  model.Query{Text: query},
		Params: model.Parameters{},
	}

	// Get tokenizer
	tokenizer, err := text_preprocessor.GetTokenizer("word")
	if err != nil {
		return nil, err
	}

	// Tokenize query
	bmx.Query.Tokens = tokenizer(query)

	// Read and process documents
	files, err := os.ReadDir(folderPath) // Changed from ioutil.ReadDir to os.ReadDir
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) == ".txt" {
			content, err := os.ReadFile(filepath.Join(folderPath, file.Name())) // Changed from ioutil.ReadFile to os.ReadFile
			if err != nil {
				return nil, err
			}

			text := string(content)
			tokens := tokenizer(text)

			doc := model.Document{
				Text:   text,
				Tokens: tokens,
			}

			bmx.Docs = append(bmx.Docs, doc)
		}
	}

	// Update N in parameters
	bmx.Params.N = len(bmx.Docs)

	// Calculate average document length
	var totalLength int
	for _, doc := range bmx.Docs {
		totalLength += len(doc.Tokens)
	}
	bmx.Params.Avgdl = float64(totalLength) / float64(len(bmx.Docs))

	bmx.Params.Alpha = max(min(1.5, bmx.Params.Avgdl/100), 0.5)
	bmx.Params.Beta = 1 / math.Log(1+float64(bmx.Params.N))

	if num_augmented_queries > 0 {
		augmentedQueries, err := GenerateAugmentedQueries(query, num_augmented_queries)
		if err != nil {
			return nil, err
		}

		for _, augmentedQuery := range augmentedQueries {
			if query != augmentedQuery {
				q := model.Query{Text: augmentedQuery}
				q.Tokens = tokenizer(q.Text)
				bmx.AugmentedQueries = append(bmx.AugmentedQueries, q)
			}
		}

		bmx.AugmentedWeights = []float64{}
		for i := 0; i < num_augmented_queries; i++ {
			bmx.AugmentedWeights = append(bmx.AugmentedWeights, 1.0/float64(num_augmented_queries+1))
		}
	}

	// Initialize BMX
	bmx.Initialize()

	return bmx, nil
}
