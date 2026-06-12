import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { Response } from 'express';
import { ChatCompletionRequestMessage } from './dto/openai.dto';

/**
 * Generation goes through OpenRouter (OpenAI-compatible). Embeddings go through
 * Google Gemini (gemini-embedding-001, 3072 dims) — its quota is separate from chat.
 */
@Injectable()
export class OpenaiService {
  private readonly GEMINI_API_KEY: string;
  private readonly OPENROUTER_API_KEY: string;

  private readonly GEMINI_EMBEDDING_URL =
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent';
  private readonly OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
  // Swap the model here — any OpenRouter chat model with tool-calling works (e.g. google/gemini-2.5-flash).
  private readonly OPENROUTER_MODEL = 'openai/gpt-4o-mini';

  constructor(private readonly configService: ConfigService) {
    this.GEMINI_API_KEY = this.configService.get('GEMINI_API_KEY') || '';
    this.OPENROUTER_API_KEY = this.configService.get('OPEN_ROUTER_API_KEY') || '';
  }

  private get openRouterHeaders() {
    return {
      Authorization: `Bearer ${this.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    };
  }

  // ---- Embeddings (Gemini) ----
  async generateGeminiEmbedding(input: string): Promise<number[]> {
    try {
      const response = await axios.post(`${this.GEMINI_EMBEDDING_URL}?key=${this.GEMINI_API_KEY}`, {
        content: { parts: [{ text: input }] },
        taskType: 'RETRIEVAL_DOCUMENT',
      });
      const embedding = response.data.embedding?.values;
      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('No embedding values returned from Gemini');
      }
      return embedding;
    } catch (error: any) {
      console.error('Gemini Embedding Error:', error?.response?.data || error?.message);
      throw new InternalServerErrorException(`Failed to generate embedding: ${error?.message}`);
    }
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
          { model: this.OPENROUTER_MODEL, messages, temperature: 0.2, stream: true },
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
        { model: this.OPENROUTER_MODEL, messages: [{ role: 'user', content: prompt }], temperature },
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
      { model: this.OPENROUTER_MODEL, messages, tools, tool_choice: 'auto', temperature: 0.2 },
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
