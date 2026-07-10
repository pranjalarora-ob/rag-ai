import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { Response } from 'express';
import { ChatCompletionRequestMessage } from './dto/openai.dto';

/**
 * Generation and Embeddings go through OpenRouter (OpenAI-compatible).
 */
@Injectable()
export class OpenaiService {
  private readonly OPENROUTER_API_KEY: string;
  private readonly OPENROUTER_MODEL: string;
  private readonly OPENROUTER_MAX_TOKENS: number;
  private readonly EMBEDDING_MODEL: string;

  private readonly OPENROUTER_EMBEDDINGS_URL = 'https://openrouter.ai/api/v1/embeddings';
  private readonly OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

  constructor(private readonly configService: ConfigService) {
    this.OPENROUTER_API_KEY = this.configService.get('OPEN_ROUTER_API_KEY') || '';
    this.OPENROUTER_MODEL = this.configService.get('OPEN_ROUTER_MODEL') || 'openai/gpt-4o-mini';
    this.OPENROUTER_MAX_TOKENS = Number(this.configService.get('OPEN_ROUTER_MAX_TOKENS')) || 800;
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

  // ---- Streaming chat (OpenRouter) ----
  async streamOpenRouter(messages: ChatCompletionRequestMessage[], res: Response) {
    return new Promise(async (resolve) => {
      let result = '';
      const errors: any[] = [];

      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      let response: any;
      try {
        response = await axios.post(
          this.OPENROUTER_URL,
          {
            model: this.OPENROUTER_MODEL,
            messages,
            temperature: 0.2,
            max_tokens: this.OPENROUTER_MAX_TOKENS,
            stream: true,
          },
          { responseType: 'stream', headers: this.openRouterHeaders },
        );
      } catch (err: any) {
        console.error('Failed to start OpenRouter stream:', err?.response?.data || err?.message);
        errors.push(err);
        if (!res.writableEnded) res.status(500).end();
        return resolve({ result, errors });
      }

      let buffer = '';
      response.data.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.replace(/^data:\s*/, '');
          if (payload === '[DONE]') {
            if (!res.writableEnded) res.end();
            return resolve({ result, errors });
          }
          try {
            const data = JSON.parse(payload);
            const content = data.choices?.[0]?.delta?.content;
            if (content) {
              res.write(content);
              (res as any).flush?.();
              result += content;
            }
          } catch {
            // ignore keep-alive comments / partial chunks
          }
        }
      });
      response.data.on('end', () => {
        if (!res.writableEnded) res.end();
        resolve({ result, errors });
      });
      response.data.on('error', (err: any) => {
        console.error('OpenRouter stream error:', err);
        errors.push(err);
        if (!res.writableEnded) res.status(500).end();
        resolve({ result, errors });
      });
      res.on('close', () => {
        if (!res.writableEnded) res.end();
        resolve({ result, errors });
      });
    });
  }

  // ---- Non-streaming text (OpenRouter) — used for routing/classification ----
  async openRouterGenerate(prompt: string, temperature = 0): Promise<string> {
    try {
      const res = await axios.post(
        this.OPENROUTER_URL,
        {
          model: this.OPENROUTER_MODEL,
          messages: [{ role: 'user', content: prompt }],
          temperature,
          max_tokens: this.OPENROUTER_MAX_TOKENS,
        },
        { headers: this.openRouterHeaders },
      );
      return res.data?.choices?.[0]?.message?.content || '';
    } catch (error: any) {
      console.error('OpenRouter generate error:', error?.response?.data || error?.message);
      return '';
    }
  }

  // ---- One tool-calling turn (OpenRouter) — returns the assistant message (may hold tool_calls) ----
  async openRouterToolTurn(messages: any[], tools: any[]): Promise<any> {
    const res = await axios.post(
      this.OPENROUTER_URL,
      {
        model: this.OPENROUTER_MODEL,
        messages,
        tools,
        tool_choice: 'auto',
        temperature: 0.2,
        max_tokens: this.OPENROUTER_MAX_TOKENS,
      },
      { headers: this.openRouterHeaders },
    );
    return res.data?.choices?.[0]?.message;
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
