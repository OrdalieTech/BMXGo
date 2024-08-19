package model

import (
	"math"
	"sort" // Added this line
	"strings"
)

// Define the parameters and types
type Document struct {
	Text   string
	Tokens []string
}

type Query struct {
	Text                  string
	Tokens                []string
	F_table               [][]float64
	IDF_table             []float64
	E_tilde_table         []float64
	max_E_tilde           float64
	E_table               []float64
	avgEntropy            float64
	S_table               []float64
	Score_table           []float64
	normalizedScore_table []float64
}

type Parameters struct {
	Alpha float64
	Beta  float64
	Avgdl float64
	N     int
}

type BMX struct {
	query                Query
	augmentedQueries     []Query
	augmentedWeights     []float64
	docs                 []Document
	params               Parameters
	augmentedScoreTable  []float64
	normalizedScoreTable []float64
}

// Function to calculate IDF
func IDF(qi string, docs []Document) float64 {
	l := 0
	for _, doc := range docs {
		if strings.Contains(doc.Text, qi) {
			l++
		}
	}
	return math.Log((float64(len(docs)-l) + 0.5) / (float64(l+1) + 0.5))
}

func (bmx *BMX) IDF_table_fill(query Query) {
	query.IDF_table = []float64{}
	for _, qi := range query.Tokens {
		query.IDF_table = append(query.IDF_table, IDF(qi, bmx.docs))
	}
}

// Function to calculate F(qi, D)
func F(qi string, doc Document) float64 {
	count := 0
	for _, token := range doc.Tokens {
		if token == qi {
			count++
		}
	}
	return float64(count)
}

func (bmx *BMX) F_table_fill(query Query) {
	query.F_table = [][]float64{}
	for _, qi := range query.Tokens {
		query.F_table = append(query.F_table, []float64{})
		for _, doc := range bmx.docs {
			query.F_table[len(query.F_table)-1] = append(query.F_table[len(query.F_table)-1], F(qi, doc))
		}
	}
}

// Function to calculate E(qi)
func E_tilde(qi string, docs []Document) float64 {
	var entropy float64
	for _, doc := range docs {
		pj := 1 / (1 + math.Exp(-F(qi, doc)))
		entropy += pj * math.Log(pj)
	}
	return -entropy
}

func (bmx *BMX) E_tilde_table_fill(query Query) {
	query.E_tilde_table = []float64{}
	for _, qi := range query.Tokens {
		query.E_tilde_table = append(query.E_tilde_table, E_tilde(qi, bmx.docs))
	}
	if query.max_E_tilde < query.E_tilde_table[len(query.E_tilde_table)-1] {
		query.max_E_tilde = query.E_tilde_table[len(query.E_tilde_table)-1]
	}
}

func (bmx *BMX) E_table_fill(query Query) {
	query.E_table = []float64{}
	for i := range query.Tokens {
		query.E_table = append(query.E_table, query.E_tilde_table[i]/query.max_E_tilde)
	}
}

func (bmx *BMX) avgEntropy(query Query) {
	query.avgEntropy = 0.0
	for i := range query.Tokens {
		query.avgEntropy += query.E_table[i]
	}
	query.avgEntropy /= float64(len(query.Tokens))
}

// Function to calculate S(Q, D)
func (bmx *BMX) S_table_fill(query Query) {
	query.S_table = []float64{}
	for i := range query.Tokens {
		for j := range bmx.docs {
			if query.F_table[i][j] > 0 {
				query.S_table = append(query.S_table, 1)
			} else {
				query.S_table = append(query.S_table, 0)
			}
		}
	}
}

// Function to calculate the score
func (bmx *BMX) Score_table_fill(query Query) {
	query.Score_table = []float64{}
	for j, doc := range bmx.docs {
		var score float64
		for i := range query.Tokens {
			idf := query.IDF_table[i]
			f := query.F_table[i][j]
			e := query.E_table[i] / query.avgEntropy
			s := query.S_table[j]
			score += idf*(f*(bmx.params.Alpha+1.0)/(f+bmx.params.Alpha*(float64(len(doc.Tokens))/bmx.params.Avgdl)+bmx.params.Alpha*query.avgEntropy)) + bmx.params.Beta*e*s
		}
		query.Score_table = append(query.Score_table, score)
	}
}

func (bmx *BMX) NormalizedScore_table_fill(query Query) {
	query.normalizedScore_table = []float64{}
	maxScore := float64(len(query.Tokens)) * (math.Log(1+float64(float64(bmx.params.N)-0.5)/1.5) + 1.0)
	for j, doc := range bmx.docs {
		var score float64
		for i := range query.Tokens {
			idf := query.IDF_table[i]
			f := query.F_table[i][j]
			e := query.E_table[i] / query.avgEntropy
			s := query.S_table[j]
			score += idf*(f*(bmx.params.Alpha+1.0)/(f+bmx.params.Alpha*(float64(len(doc.Tokens))/bmx.params.Avgdl)+bmx.params.Alpha*query.avgEntropy)) + bmx.params.Beta*e*s
		}
		query.normalizedScore_table = append(query.normalizedScore_table, score/maxScore)
	}
}

// Function to calculate the weighted query augmentation score
func (bmx *BMX) AugmentedScore() {
	bmx.augmentedScoreTable = []float64{}
	for j := range bmx.docs {
		score := bmx.query.Score_table[j]
		for i, Q_A := range bmx.augmentedQueries {
			score += bmx.augmentedWeights[i] * Q_A.Score_table[j]
		}
		bmx.augmentedScoreTable = append(bmx.augmentedScoreTable, score)
	}
}

func (bmx *BMX) NormalizedAugmentedScore() {
	bmx.normalizedScoreTable = []float64{}
	for j := range bmx.docs {
		score := bmx.query.normalizedScore_table[j]
		for i, Q_A := range bmx.augmentedQueries {
			score += bmx.augmentedWeights[i] * Q_A.normalizedScore_table[j]
		}
		bmx.normalizedScoreTable = append(bmx.normalizedScoreTable, score)
	}
}

func (bmx *BMX) InitializeQuery(query Query) {
	bmx.IDF_table_fill(bmx.query)
	bmx.F_table_fill(bmx.query)
	bmx.E_tilde_table_fill(bmx.query)
	bmx.E_table_fill(bmx.query)
	bmx.avgEntropy(bmx.query)
	bmx.S_table_fill(bmx.query)
	bmx.Score_table_fill(bmx.query)
	bmx.NormalizedScore_table_fill(bmx.query)
}

func (bmx *BMX) Initialize() {
	bmx.InitializeQuery(bmx.query)
	for _, q := range bmx.augmentedQueries {
		bmx.InitializeQuery(q)
	}
	bmx.AugmentedScore()
	bmx.NormalizedAugmentedScore()
}

func (bmx *BMX) Rank() []int {
	// Create a slice of indices
	indices := make([]int, len(bmx.normalizedScoreTable))
	for i := range indices {
		indices[i] = i
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(indices, func(i, j int) bool {
		return bmx.normalizedScoreTable[indices[i]] > bmx.normalizedScoreTable[indices[j]]
	})

	return indices
}
