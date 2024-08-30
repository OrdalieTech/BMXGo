# BMXGo

BMXGo is a Go implementation of the BMX (Best Matching X) algorithm for information retrieval and search. This project provides a flexible and efficient search engine that can be easily integrated into various applications.

## Features

- Efficient text preprocessing and tokenization
- Support for multiple languages
- Customizable stopword removal and stemming
- Query augmentation for improved search results
- Concurrent processing for better performance

## Installation

To use BMXGo in your project, you need to have Go 1.22.5 or later installed. Then, you can install the package using:

```bash
go get github.com/OrdalieTech/BMXGo
```

## Usage

Here's a simple example to get you started:

```go
package main

import (
	"fmt"
	"github.com/OrdalieTech/BMXGo"
)

func main() {
    // Create a new BMX adapter
    config := BMXGo.text_preprocessor.Config{
    // Configure your text preprocessing options here
    }
    adapter := BMXGo.Build("my_index", config)
    // Add documents to the index
    ids := []string{"doc1", "doc2", "doc3"}
    docs := []string{
        "This is the first document",
        "This is the second document",
        "And this is the third one",
    }
    adapter.AddMany(ids, docs)
    // Perform a search
    query := "second document"
    results := adapter.Search(query, 3)
    // Print the results
    for i, key := range results.Keys {
        fmt.Printf("Result %d: %s (Score: %.4f)\n", i+1, key, results.Scores[i])
    }
}
```

## Advanced Usage

### Query Augmentation

BMXGo supports query augmentation to improve search results:

```go
results := adapter.SearchAugmented(query, 10, 3, 0.5)
```

This will generate 3 augmented queries with a weight of 0.5 each.

### Concurrent Processing

For better performance with large datasets, use the concurrent processing methods:

```go
results := adapter.SearchAugmentedMany(queries, 10, 3, 0.5, 50)
```

This will process multiple queries concurrently using 50 workers.

## Configuration

You can customize the text preprocessing pipeline by modifying the `Config` struct:

```go
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
```

## Dependencies

BMXGo relies on the following external packages:

```go
github.com/PuerkitoBio/goquery v1.9.2
github.com/kljensen/snowball v0.10.0
github.com/rainycape/unidecode v0.0.0-20150907023854-cb7f23ec59be
github.com/reiver/go-porterstemmer v1.0.1
golang.org/x/text v0.14.0
github.com/andybalholm/cascadia v1.3.2 // indirect
golang.org/x/net v0.24.0
```

## Contributing

Contributions to BMXGo are welcome! Please feel free to submit a Pull Request.


