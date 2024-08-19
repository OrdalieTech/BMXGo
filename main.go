package main

import (
	"fmt"
	"strings"
)

func main() {
	text := "This is a sample text to tokenize."
	tokens := strings.Fields(text)
	fmt.Println(tokens)
}
