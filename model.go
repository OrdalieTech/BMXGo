package BMXGo

import (
	"math"
	"sort"
)

// Define the parameters and types
type Document struct {
	Text    string
	Tokens  []string
	F_table map[string]int
}

type Query struct {
	Text                 string
	Tokens               map[string]float64
	TotalWeight          float64
	max_E_tilde          float64
	avgEntropy           float64
	S_table              map[string]float64
	ScoreTable           map[string]float64
	NormalizedScoreTable map[string]float64
	AugmentedQueries     []string
	AugmentedWeights     []float64
}

type Parameters struct {
	Alpha float64
	Beta  float64
	Avgdl float64
	N     int
}

type BMX struct {
	Docs             map[string]Document
	Params           Parameters
	TextPreprocessor *TextPreprocessor
	NumAppearances   map[string][]string
	IDF_table        map[string]float64
	E_tilde_table    map[string]float64
}

func (bmx *BMX) InitializeTextPreprocessor(config *Config) error {
	tp := NewTextPreprocessor(config)
	bmx.TextPreprocessor = tp
	return nil
}

func (bmx *BMX) SetParams() {
	N := len(bmx.Docs)
	var totalLength int
	for _, doc := range bmx.Docs {
		totalLength += len(doc.Tokens)
	}
	Avgdl := float64(totalLength) / float64(N)
	Alpha := max(min(1.5, Avgdl/100), 0.5)
	Beta := 1 / math.Log(1+float64(N))

	bmx.Params = Parameters{
		Alpha: Alpha,
		Beta:  Beta,
		Avgdl: Avgdl,
		N:     N,
	}
}

func (bmx *BMX) F_table_fill() {
	for doc_key := range bmx.Docs {
		doc := bmx.Docs[doc_key]
		doc.F_table = make(map[string]int)
		for _, token := range doc.Tokens {
			if _, ok := doc.F_table[token]; !ok {
				doc.F_table[token] = 1
			} else {
				doc.F_table[token]++
			}
		}
		bmx.Docs[doc_key] = doc
	}
}

func (bmx *BMX) NumAppearancesCalc() {
	bmx.NumAppearances = make(map[string][]string)
	for doc_key, doc := range bmx.Docs {
		for token := range doc.F_table {
			if _, ok := bmx.NumAppearances[token]; !ok {
				bmx.NumAppearances[token] = []string{doc_key}
			} else {
				bmx.NumAppearances[token] = append(bmx.NumAppearances[token], doc_key)
			}
		}
	}
}

func (bmx *BMX) IDF_table_fill() {
	bmx.IDF_table = make(map[string]float64)
	for token := range bmx.NumAppearances {
		bmx.IDF_table[token] = math.Log((float64(bmx.Params.N-len(bmx.NumAppearances[token]))+0.5)/(float64(len(bmx.NumAppearances[token]))+0.5) + 1.0)
	}
}

func (bmx *BMX) E_tilde_table_fill() {
	bmx.E_tilde_table = make(map[string]float64)
	for _, doc := range bmx.Docs {
		for _, qi := range doc.Tokens {
			pj := 1 / (1 + math.Exp(float64(-doc.F_table[qi])))
			if _, ok := bmx.E_tilde_table[qi]; !ok {
				bmx.E_tilde_table[qi] = -pj * math.Log(pj)
			} else {
				bmx.E_tilde_table[qi] += -pj * math.Log(pj)
			}
		}
	}
}

func (query *Query) SetEntropy(bmx *BMX) {
	query.max_E_tilde = 0.0
	query.avgEntropy = 0.0
	for qi := range query.Tokens {
		if bmx.E_tilde_table[qi] > query.max_E_tilde {
			query.max_E_tilde = bmx.E_tilde_table[qi]
		}
		query.avgEntropy += bmx.E_tilde_table[qi] * query.Tokens[qi]
	}
	query.avgEntropy /= query.TotalWeight
	query.avgEntropy /= query.max_E_tilde
}

// Function to calculate S(Q, D)
func (query *Query) S_table_fill(bmx *BMX) {
	query.S_table = make(map[string]float64, len(bmx.Docs))
	invTotalWeight := 1.0 / query.TotalWeight

	for doc_key := range bmx.Docs {
		query.S_table[doc_key] = 0.0
	}

	for qi, weight := range query.Tokens {
		for _, doc_key := range bmx.NumAppearances[qi] {
			query.S_table[doc_key] += weight
		}
	}
	for doc_key := range bmx.Docs {
		query.S_table[doc_key] *= invTotalWeight
	}
}

// Function to calculate the score
func (query *Query) Score_table_fill(bmx *BMX) {
	query.ScoreTable = make(map[string]float64, len(bmx.Docs))
	for doc_key := range bmx.Docs {
		query.ScoreTable[doc_key] = 0.0
	}
	invE_tilde := 1.0 / query.max_E_tilde
	invAvgdl := 1.0 / bmx.Params.Avgdl
	alpha1 := bmx.Params.Alpha + 1.0
	for qi := range query.Tokens {
		idf := bmx.IDF_table[qi]
		e := bmx.E_tilde_table[qi] * invE_tilde
		alphaAverageEntropy := bmx.Params.Alpha * query.avgEntropy
		betaE := bmx.Params.Beta * e
		for _, doc_key := range bmx.NumAppearances[qi] {
			f := bmx.Docs[doc_key].F_table[qi]
			s := query.S_table[doc_key]
			query.ScoreTable[doc_key] += query.Tokens[qi] * (idf*(float64(f)*alpha1/(float64(f)+bmx.Params.Alpha*(float64(len(bmx.Docs[doc_key].Tokens))*invAvgdl)+alphaAverageEntropy)) + betaE*s)
		}
	}
}

func (query *Query) NormalizedScore_table_fill(bmx *BMX) {
	query.NormalizedScoreTable = map[string]float64{}
	invMaxScore := 1 / (query.TotalWeight * (math.Log(1+float64(float64(bmx.Params.N)-0.5)/1.5) + 1.0))
	for key := range bmx.Docs {
		query.NormalizedScoreTable[key] = query.ScoreTable[key] * invMaxScore
	}
}

func (query *Query) Initialize(bmx *BMX) {
	tokens := bmx.TextPreprocessor.Process(query.Text)
	query.Tokens = make(map[string]float64)
	for _, token := range tokens {
		if _, ok := query.Tokens[token]; !ok {
			query.Tokens[token] = 1.0
		} else {
			query.Tokens[token] += 1.0
		}
		query.TotalWeight += 1.0
	}
	for i := range query.AugmentedQueries {
		tokens := bmx.TextPreprocessor.Process(query.AugmentedQueries[i])
		for _, token := range tokens {
			if _, ok := query.Tokens[token]; !ok {
				query.Tokens[token] = query.AugmentedWeights[i]
			} else {
				query.Tokens[token] += query.AugmentedWeights[i]
			}
			query.TotalWeight += query.AugmentedWeights[i]
		}
	}
	// fmt.Println("Setting entropy")
	// start := time.Now()
	query.SetEntropy(bmx)
	// fmt.Println("Entropy set, total time:", time.Since(start))
	// fmt.Println("Setting S table")
	// start = time.Now()
	query.S_table_fill(bmx)
	// fmt.Println("S table set, total time:", time.Since(start))
	// fmt.Println("Setting Score table")
	// start = time.Now()
	query.Score_table_fill(bmx)
	// fmt.Println("Score table filled, total time:", time.Since(start))
	// fmt.Println("Setting Normalized Score table")
	// start = time.Now()
	query.NormalizedScore_table_fill(bmx)
	// fmt.Println("Normalized score table filled, total time:", time.Since(start))
}

func (query *Query) Rank(topK int) []string {
	// Create a slice of indices
	topKeys := make([]string, len(query.ScoreTable))
	for key := range query.ScoreTable {
		topKeys = append(topKeys, key)
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(topKeys, func(i, j int) bool {
		return query.ScoreTable[topKeys[i]] > query.ScoreTable[topKeys[j]]
	})
	return topKeys[:topK]
}
