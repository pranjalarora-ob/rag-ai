import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

/**
 * Embeddings go through OpenRouter (OpenAI-compatible). All LLM/chat work is in ClaudeService.
 */
@Injectable()
export class OpenaiService {
  private readonly OPENROUTER_API_KEY: string;
  private readonly EMBEDDING_MODEL: string;

  private readonly OPENROUTER_EMBEDDINGS_URL = 'https://openrouter.ai/api/v1/embeddings';

  constructor(private readonly configService: ConfigService) {
    this.OPENROUTER_API_KEY = this.configService.get('OPEN_ROUTER_API_KEY') || '';
    this.EMBEDDING_MODEL = this.configService.get('EMBEDDING_MODEL') || 'openai/text-embedding-3-small';
  }

  private get openRouterHeaders() {
    return {
      Authorization: `Bearer ${this.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    };
  }

  // ---- Embeddings (OpenRouter/OpenAI) ----
  async generateEmbedding(input: string): Promise<number[]> {
    try {
      const response = await axios.post(
        this.OPENROUTER_EMBEDDINGS_URL,
        {
          model: this.EMBEDDING_MODEL,
          input,
        },
        {
          headers: this.openRouterHeaders,
        },
      );
      const embedding = response.data.data?.[0]?.embedding;
      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('No embedding values returned from OpenRouter');
      }
      return embedding;
    } catch (error: any) {
      console.error('OpenRouter Embedding Error:', error?.response?.data || error?.message);
      throw new InternalServerErrorException(`Failed to generate embedding: ${error?.message}`);
    }
  }

  // ---- Batched embeddings — one HTTP call per `batchSize` texts (used by ingestion) ----
  async generateEmbeddings(inputs: string[], batchSize = 96): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < inputs.length; i += batchSize) {
      const batch = inputs.slice(i, i + batchSize);
      try {
        const response = await axios.post(
          this.OPENROUTER_EMBEDDINGS_URL,
          { model: this.EMBEDDING_MODEL, input: batch },
          { headers: this.openRouterHeaders },
        );
        const data = response.data?.data;
        if (!Array.isArray(data) || data.length !== batch.length) {
          throw new Error('Embedding count mismatch from provider');
        }
        // The API may return items out of order — sort by `index` before collecting.
        [...data].sort((a, b) => a.index - b.index).forEach((d) => out.push(d.embedding));
      } catch (error: any) {
        console.error('OpenRouter Batch Embedding Error:', error?.response?.data || error?.message);
        throw new InternalServerErrorException(`Failed to generate embeddings: ${error?.message}`);
      }
    }
    return out;
  }

  // Greedily join chunks up to an approximate token budget (no tokenizer dependency).
  async truncateByTokens(texts: string[], maxTokens = Infinity) {
    const selected: string[] = [];
    let total = 0;
    for (const text of texts) {
      const tokens = Math.ceil((text?.split(/\s+/).length || 0) * 1.3);
      if (total + tokens > maxTokens) break;
      selected.push(text);
      total += tokens;
    }
    return selected.join('\n');
  }
}
