// Google AI (Gemini) — `GOOGLE_API_KEY` in repo `.env` (loaded before inference runs)
client<llm> CustomGemini {
  provider google-ai
  retry_policy GeminiQuotaRetry
  options {
    // Swap model here or via a second client block (BAML does not read model from env)
    model "gemini-2.5-flash"
    api_key env.GOOGLE_API_KEY
    // thinkingConfig must live inside generationConfig for generateContent (not top-level)
    generationConfig {
      temperature 0
      max_output_tokens 8192
      thinkingConfig {
        thinking_budget 2048
      }
    }
  }
}

// Free-tier Gemini 429 responses often include RetryInfo ~60s; wait long enough between retries.
retry_policy GeminiQuotaRetry {
  max_retries 5
  strategy {
    type exponential_backoff
    delay_ms 5000
    multiplier 2.0
    max_delay_ms 65000
  }
}