package build

import (
	"os"
	"path/filepath"

	model "BMXGo/search/model"

	text_preprocessor "../text_preprocessor"
)

// BuildBMX creates a BMX object from a query and a folder of text documents
func BuildBMX(query string, folderPath string, params model.Parameters) (*model.BMX, error) {
	// Initialize BMX struct
	bmx := &model.BMX{
		query:  model.Query{Text: query},
		params: params,
	}

	// Get tokenizer
	tokenizer, err := text_preprocessor.GetTokenizer("word")
	if err != nil {
		return nil, err
	}

	// Tokenize query
	bmx.query.Tokens = tokenizer(query)

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

			bmx.docs = append(bmx.docs, doc)
		}
	}

	// Update N in parameters
	bmx.params.N = len(bmx.docs)

	// Calculate average document length
	var totalLength int
	for _, doc := range bmx.docs {
		totalLength += len(doc.Tokens)
	}
	bmx.params.Avgdl = float64(totalLength) / float64(len(bmx.docs))

	// Initialize BMX
	bmx.Initialize()

	return bmx, nil
}
