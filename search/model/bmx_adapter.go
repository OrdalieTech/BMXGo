package model

import (
	"BMXGo/search/text_preprocessor"
)

type BMXAdapter struct {
	bmx *BMX
}

func NewBMXAdapter(docs []Document, params Parameters) *BMXAdapter {
	bmx := &BMX{
		Docs:   docs,
		Params: params,
	}
	return &BMXAdapter{bmx: bmx}
}

func (adapter *BMXAdapter) Search(query string, topK int) []int {
	tokenize, _ := text_preprocessor.GetTokenizer("word")
	adapter.bmx.Query = Query{
		Text:   query,
		Tokens: tokenize(query),
	}
	adapter.bmx.Initialize()
	rankedDocs := adapter.bmx.Rank(topK)

	// Ensure we return at most topK results
	if len(rankedDocs) > topK {
		rankedDocs = rankedDocs[:topK]
	}
	return rankedDocs
}

func (adapter *BMXAdapter) SearchAugmented(query string, augmentedQueries []string, topK int) []int {
	tokenize, _ := text_preprocessor.GetTokenizer("word")
	adapter.bmx.Query = Query{
		Text:   query,
		Tokens: tokenize(query),
	}

	adapter.bmx.AugmentedQueries = make([]Query, len(augmentedQueries))
	for i, augQuery := range augmentedQueries {
		adapter.bmx.AugmentedQueries[i] = Query{
			Text:   augQuery,
			Tokens: tokenize(augQuery),
		}
	}

	adapter.bmx.Initialize()
	rankedDocs := adapter.bmx.RankAugmented(topK)

	// Ensure we return at most topK results
	if len(rankedDocs) > topK {
		rankedDocs = rankedDocs[:topK]
	}
	return rankedDocs
}
