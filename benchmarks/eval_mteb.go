package main

import (
	"BMXGo/search/model"
	"BMXGo/search/text_preprocessor"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

func eval_mteb() {
	if len(os.Args) != 3 {
		fmt.Println("usage: go run eval_mteb.go model dataset")
		os.Exit(1)
	}

	modelName := os.Args[1]
	datasetName := os.Args[2]
	splitName := "test"
	if datasetName == "msmarco" {
		splitName = "dev"
	}

	fmt.Printf("model_name=%s\n", modelName)
	fmt.Printf("dataset_name=%s\n", datasetName)
	fmt.Printf("split_name=%s\n", splitName)

	// Load dataset
	folderPath := filepath.Join("Test", "textes_extraits")

	// Prepare documents
	var docs []model.Document
	files, _ := os.ReadDir(folderPath)
	tokenize, _ := text_preprocessor.GetTokenizer("word")
	for _, file := range files {
		content, _ := os.ReadFile(filepath.Join(folderPath, file.Name()))
		text := string(content)
		docs = append(docs, model.Document{
			Text:   text,
			Tokens: tokenize(text),
		})
	}

	// Initialize BMXAdapter
	params := model.Parameters{
		Alpha: 1.2,
		Beta:  0.75,
		Avgdl: 100, // You may want to calculate this based on your dataset
		N:     len(docs),
	}
	bmxAdapter := model.NewBMXAdapter(docs, params)

	// Prepare queries
	queries := []string{
		"Réduction du Temps de Travail",
		"congé de maternité",
		"manutention manuelle",
	}

	// Search
	fmt.Println("Searching...")
	startTime := time.Now()
	for _, query := range queries {
		rankedDocs := bmxAdapter.Search(query, 5)
		fmt.Printf("Query: %s\n", query)
		for i, docIndex := range rankedDocs {
			fmt.Printf("  %d. %s\n", i+1, docs[docIndex].Text[:100])
		}
		fmt.Println()
	}
	fmt.Printf("Search time: %v\n", time.Since(startTime))

	// TODO: Implement evaluation metrics (NDCG, MAP, Recall, Precision)
}
