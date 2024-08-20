package main

import (
	"BMXGo/search/build"
	"fmt"
)

func main() {
	query := "réduction du temps de travail en CDI"
	folderPath := "Test/textes_extraits"
	num_augmented_queries := 3
	bmx, err := build.BuildBMX(query, folderPath, num_augmented_queries)
	if err != nil {
		fmt.Println("Erreur lors de la construction de BMX:", err)
		return
	}

	l := bmx.Rank(5)

	for _, i := range l {
		fmt.Println(bmx.Docs[i].Text)
		fmt.Println(bmx.AugmentedScoreTable[i])
		fmt.Println(bmx.NormalizedScoreTable[i])
		fmt.Println("BREAK")
	}
}
