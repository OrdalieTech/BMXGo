package build

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

type LinkedContent struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

type ConvMessage struct {
	Role          string          `json:"role"`
	Content       string          `json:"content"`
	Date          time.Time       `json:"date,omitempty"`
	Intent        string          `json:"intent,omitempty"`
	LinkedContent []LinkedContent `json:"linkedContent,omitempty"`
}

type LLMClient struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	appName    string
	appURL     string
}

type ClientConfig struct {
	APIKey         string
	BaseURL        string
	HTTPClient     *http.Client
	AppName        string
	AppURL         string
	Provider       string
	ResourceName   string
	DeploymentName string
}

func HtmlToMarkdown(htmlContent string, addIDs bool) string {
	replaceFormattingTags := func(html string) string {
		html = strings.ReplaceAll(html, "<strong>", "<strong>**")
		html = strings.ReplaceAll(html, "</strong>", "**</strong>")
		html = strings.ReplaceAll(html, "<em>", "<em>*")
		html = strings.ReplaceAll(html, "</em>", "*</em>")
		return html
	}
	htmlContent = replaceFormattingTags(htmlContent)

	doc, err := goquery.NewDocumentFromReader(strings.NewReader(htmlContent))
	if err != nil {
		return ""
	}

	markdown := []string{}
	seenTexts := map[string]bool{}
	sectionIndex := 1

	doc.Find("header, nav, footer, meta, script, style, link, noscript, head").Each(func(i int, s *goquery.Selection) {
		s.Remove()
	})

	breadcrumbSelectors := []string{
		"div.breadcrumb", "ul.breadcrumb", "nav[aria-label=\"breadcrumb\"]",
		"ol.breadcrumb", "[id*=\"breadcrumb\"]", "[class*=\"breadcrumb\"]",
		"[id*=\"fil-ariane\"]", "[class*=\"fil-ariane\"]",
	}
	for _, selector := range breadcrumbSelectors {
		doc.Find(selector).Each(func(i int, s *goquery.Selection) {
			s.Remove()
		})
	}

	doc.Find("[aria-hidden=\"true\"], .btn").Each(func(i int, s *goquery.Selection) {
		s.Remove()
	})

	doc.Find("button").Each(func(i int, s *goquery.Selection) {
		s.Remove()
	})

	addToMarkdown := func(text string, tag *goquery.Selection) {
		if text == "" {
			return
		}
		seenTextSample := strings.TrimSpace(text)
		if len(seenTextSample) > 2000 {
			seenTextSample = seenTextSample[:2000]
		}
		if !seenTexts[seenTextSample] {
			if addIDs && tag != nil {
				text = "【¶" + strconv.Itoa(sectionIndex) + "】 " + text
				tag.SetAttr("id", "【¶"+strconv.Itoa(sectionIndex)+"】")
			}
			markdown = append(markdown, text, "")
			seenTexts[seenTextSample] = true
		}
	}

	shouldSkipList := func(tag *goquery.Selection) bool {
		if tag.Is("ul, ol") {
			linkCount := tag.Find("a, button").Length()
			return linkCount >= 2
		}
		return false
	}

	doc.Find("h1, h2, h3, h4, h5, h6, p, ul, ol, li, blockquote, table, br, div").Each(func(i int, tag *goquery.Selection) {
		if shouldSkipList(tag) {
			return
		}
		switch goquery.NodeName(tag) {
		case "h1", "h2", "h3", "h4", "h5", "h6":
			addToMarkdown(strings.Repeat("#", len(goquery.NodeName(tag))-1)+" "+tag.Text(), tag)
		case "a":
			if href, ok := tag.Attr("href"); ok && strings.HasPrefix(href, "http") {
				addToMarkdown(fmt.Sprintf("[%s](%s)", tag.Text(), href), tag)
			}
		case "p":
			addToMarkdown(tag.Text(), tag)
		case "ul", "ol":
			tag.Find("li").Each(func(i int, li *goquery.Selection) {
				prefix := "*"
				if goquery.NodeName(tag) == "ol" {
					prefix = "1."
				}
				addToMarkdown(prefix+" "+li.Text(), li)
			})
		case "blockquote":
			lines := strings.Split(tag.Text(), "\n")
			for _, line := range lines {
				if strings.TrimSpace(line) != "" {
					addToMarkdown("> "+strings.TrimSpace(line), tag)
				}
			}
		case "table":
			markdown = append(markdown, tag.Text(), "")
		case "br":
			markdown = append(markdown, "\n")
		case "div":
			if class, _ := tag.Attr("class"); strings.Contains(strings.ToLower(class), "article") || strings.Contains(strings.ToLower(class), "content") {
				if tag.Find("p, h1, h2, h3, h4, h5, h6").Length() == 0 {
					addToMarkdown(tag.Text(), tag)
				}
			}
		}
		if addIDs {
			sectionIndex++
		}
	})

	return strings.TrimSpace(strings.Join(markdown, "\n"))
}

func NewLLMClient(config ClientConfig) *LLMClient {
	switch config.Provider {
	case "azure":
		config.BaseURL = fmt.Sprintf("https://%s/openai/deployments/%s/chat/completions?api-version=2023-12-01-preview", os.Getenv("AZURE_OAI_DOMAIN"), config.DeploymentName)
		config.APIKey = os.Getenv("AZURE_API_KEY")
	case "openai":
		config.BaseURL = "https://api.openai.com/v1/chat/completions"
		config.APIKey = os.Getenv("OPENAI_API_KEY")
	case "openrouter":
		config.BaseURL = "https://openrouter.ai/api/v1/chat/completions"
		config.APIKey = os.Getenv("OPENROUTER_API_KEY")
	default:
		config.BaseURL = "https://openrouter.ai/api/v1/chat/completions"
		config.APIKey = os.Getenv("OPENROUTER_API_KEY")
	}

	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{Timeout: 120 * time.Second}
	}

	return &LLMClient{
		apiKey:     config.APIKey,
		baseURL:    config.BaseURL,
		httpClient: config.HTTPClient,
		appName:    config.AppName,
		appURL:     config.AppURL,
	}
}

type ChatCompletionRequest struct {
	Models      []string      `json:"models"`
	Messages    []ConvMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature float32       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

func (c *LLMClient) Completion(ctx context.Context, request ChatCompletionRequest) (<-chan string, <-chan error) {
	responseChan := make(chan string)
	errChan := make(chan error)

	go func() {
		defer close(responseChan)
		defer close(errChan)

		for _, model := range request.Models {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
				requestBody := map[string]interface{}{
					"model":       model,
					"messages":    request.Messages,
					"stream":      request.Stream,
					"temperature": request.Temperature,
					"max_tokens":  4000,
				}

				jsonBody, err := json.Marshal(requestBody)
				if err != nil {
					errChan <- fmt.Errorf("error marshaling request for model %s: %v", model, err)
					continue
				}
				req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL, bytes.NewReader(jsonBody))
				if err != nil {
					errChan <- fmt.Errorf("error creating request for model %s: %v", model, err)
					continue
				}

				req.Header.Set("Authorization", "Bearer "+c.apiKey)
				req.Header.Set("api-key", c.apiKey)
				req.Header.Set("Content-Type", "application/json")

				resp, err := c.httpClient.Do(req)
				if err != nil {
					errChan <- fmt.Errorf("error making request for model %s: %v", model, err)
					continue
				}
				defer resp.Body.Close()

				if resp.StatusCode != http.StatusOK {
					body, _ := io.ReadAll(resp.Body)
					errChan <- fmt.Errorf("model %s failed with status code: %d\nResponse body: %s", model, resp.StatusCode, string(body))
					continue
				}

				if request.Stream {
					c.handleStreamingResponse(resp.Body, responseChan, errChan)
				} else {
					c.handleNonStreamingResponse(resp.Body, responseChan, errChan)
				}
				return
			}
		}

		errChan <- fmt.Errorf("all models failed")
	}()

	return responseChan, errChan
}

func (c *LLMClient) handleStreamingResponse(body io.ReadCloser, responseChan chan<- string, errChan chan<- error) {
	reader := bufio.NewReader(body)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				errChan <- fmt.Errorf("error reading stream: %v", err)
			}
			return
		}

		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		line = bytes.TrimPrefix(line, []byte("data: "))

		if string(line) == "[DONE]" {
			return
		}

		var streamResponse struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal(line, &streamResponse); err != nil {
			errChan <- err
			return
		}

		if len(streamResponse.Choices) > 0 {
			responseChan <- streamResponse.Choices[0].Delta.Content
		}
	}
}

func (c *LLMClient) handleNonStreamingResponse(body io.ReadCloser, responseChan chan<- string, errChan chan<- error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(body).Decode(&response); err != nil {
		errChan <- fmt.Errorf("error decoding response: %v", err)
		return
	}

	if len(response.Choices) > 0 {
		responseChan <- response.Choices[0].Message.Content
	} else {
		errChan <- fmt.Errorf("no content in response")
	}
}
