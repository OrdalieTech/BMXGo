package BMXGo

import (
	"math"
	"sort"
)

// Define the parameters and types
type Document struct {
	Text   string
	Tokens []string
}

type Query struct {
	Text                  string
	Tokens                []string
	F_table               []map[string]float64
	IDF_table             []float64
	E_tilde_table         []float64
	max_E_tilde           float64
	E_table               []float64
	avgEntropy            float64
	S_table               map[string]float64
	Score_table           map[string]float64
	NormalizedScore_table map[string]float64
}

type Parameters struct {
	Alpha float64
	Beta  float64
	Avgdl float64
	N     int
}

type BMX struct {
	Query                Query
	AugmentedQueries     []Query
	AugmentedWeights     []float64
	Docs                 map[string]Document
	Params               Parameters
	AugmentedScoreTable  map[string]float64
	NormalizedScoreTable map[string]float64
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

// Function to calculate IDF
func IDF(qi string, docs map[string]Document) float64 {
	l := 0
	for _, doc := range docs {
		for _, token := range doc.Tokens {
			if token == qi {
				l++
				break
			}
		}
	}
	return math.Log((float64(len(docs)-l)+0.5)/(float64(l)+0.5) + 1.0)
}

func (bmx *BMX) IDF_table_fill(query *Query) {
	query.IDF_table = []float64{}
	for _, qi := range query.Tokens {
		query.IDF_table = append(query.IDF_table, IDF(qi, bmx.Docs))
	}
}

// Function to calculate F(qi, D)
func F(qi string, doc *Document) float64 {
	count := 0
	for _, token := range doc.Tokens {
		if token == qi {
			count++
		}
	}
	return float64(count)
}

func (bmx *BMX) F_table_fill(query *Query) {
	query.F_table = []map[string]float64{}
	for _, qi := range query.Tokens {
		query.F_table = append(query.F_table, map[string]float64{})
		for key, doc := range bmx.Docs {
			query.F_table[len(query.F_table)-1][key] = F(qi, &doc)
		}
	}
}

// Function to calculate E(qi)
func E_tilde(qi string, docs map[string]Document) float64 {
	var entropy float64
	for _, doc := range docs {
		v := -F(qi, &doc)
		if v != 0 {
			pj := 1 / (1 + math.Exp(v))
			entropy += pj * math.Log(pj)
		}
	}
	return -entropy
}

func (bmx *BMX) E_tilde_table_fill(query *Query) {
	query.E_tilde_table = []float64{}
	for _, qi := range query.Tokens {
		query.E_tilde_table = append(query.E_tilde_table, E_tilde(qi, bmx.Docs))
		if query.max_E_tilde < query.E_tilde_table[len(query.E_tilde_table)-1] {
			query.max_E_tilde = query.E_tilde_table[len(query.E_tilde_table)-1]
		}
	}
}

func (bmx *BMX) E_table_fill(query *Query) {
	query.E_table = []float64{}
	for i := range query.Tokens {
		query.E_table = append(query.E_table, query.E_tilde_table[i]/query.max_E_tilde)
	}
}

func (bmx *BMX) avgEntropy(query *Query) {
	query.avgEntropy = 0.0
	for i := range query.Tokens {
		query.avgEntropy += query.E_table[i]
	}
	query.avgEntropy /= float64(len(query.Tokens))
}

// Function to calculate S(Q, D)
func (bmx *BMX) S_table_fill(query *Query) {
	query.S_table = map[string]float64{}
	for key := range bmx.Docs {
		query.S_table[key] = 0
		for i := range query.Tokens {
			if query.F_table[i][key] > 0 {
				query.S_table[key] += 1
			}
		}
	}
}

// Function to calculate the score
func (bmx *BMX) Score_table_fill(query *Query) {
	query.Score_table = map[string]float64{}
	for key, doc := range bmx.Docs {
		var score float64
		for i := range query.Tokens {
			idf := query.IDF_table[i]
			f := query.F_table[i][key]
			e := query.E_table[i]
			s := query.S_table[key]
			score += idf*(f*(bmx.Params.Alpha+1.0)/(f+bmx.Params.Alpha*(float64(len(doc.Tokens))/bmx.Params.Avgdl)+bmx.Params.Alpha*query.avgEntropy)) + bmx.Params.Beta*e*s
		}
		query.Score_table[key] = score
	}
}

func (bmx *BMX) NormalizedScore_table_fill(query *Query) {
	query.NormalizedScore_table = map[string]float64{}
	maxScore := float64(len(query.Tokens)) * (math.Log(1+float64(float64(bmx.Params.N)-0.5)/1.5) + 1.0)
	for key := range bmx.Docs {
		query.NormalizedScore_table[key] = query.Score_table[key] / maxScore
	}
}

// Function to calculate the weighted query augmentation score
func (bmx *BMX) AugmentedScore() {
	bmx.AugmentedScoreTable = map[string]float64{}
	for key := range bmx.Docs {
		score := bmx.Query.Score_table[key]
		for i, Q_A := range bmx.AugmentedQueries {
			score += bmx.AugmentedWeights[i] * Q_A.Score_table[key]
		}
		bmx.AugmentedScoreTable[key] = score
	}
}

func (bmx *BMX) NormalizedAugmentedScore() {
	bmx.NormalizedScoreTable = map[string]float64{}
	for key := range bmx.Docs {
		score := bmx.Query.NormalizedScore_table[key]
		for i, Q_A := range bmx.AugmentedQueries {
			score += bmx.AugmentedWeights[i] * Q_A.NormalizedScore_table[key]
		}
		bmx.NormalizedScoreTable[key] = score
	}
}

func (bmx *BMX) InitializeQuery(query *Query) {
	bmx.IDF_table_fill(query)
	bmx.F_table_fill(query)
	bmx.E_tilde_table_fill(query)
	bmx.E_table_fill(query)
	bmx.avgEntropy(query)
	bmx.S_table_fill(query)
	bmx.Score_table_fill(query)
	bmx.NormalizedScore_table_fill(query)
}

func (bmx *BMX) Initialize() {
	bmx.SetParams()
	bmx.InitializeQuery(&bmx.Query)
	for i := range bmx.AugmentedQueries {
		bmx.InitializeQuery(&bmx.AugmentedQueries[i])
	}
	bmx.AugmentedScore()
	bmx.NormalizedAugmentedScore()
}

func (bmx *BMX) Rank(topK int) []string {
	// Create a slice of indices
	topKeys := make([]string, len(bmx.Query.Score_table))
	for key := range bmx.Query.Score_table {
		topKeys = append(topKeys, key)
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(topKeys, func(i, j int) bool {
		return bmx.Query.Score_table[topKeys[i]] > bmx.Query.Score_table[topKeys[j]]
	})
	return topKeys[:topK]
}

func (bmx *BMX) RankAugmented(topK int) []string {
	// Create a slice of indices
	topKeys := make([]string, len(bmx.AugmentedScoreTable))
	for key := range bmx.AugmentedScoreTable {
		topKeys = append(topKeys, key)
	}

	// Sort the indices based on the normalizedScoreTable in descending order
	sort.Slice(topKeys, func(i, j int) bool {
		return bmx.AugmentedScoreTable[topKeys[i]] > bmx.AugmentedScoreTable[topKeys[j]]
	})
	return topKeys[:topK]
}
