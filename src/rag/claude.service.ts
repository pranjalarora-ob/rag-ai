import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { Response } from 'express';
import { ChatCompletionRequestMessage } from './dto/openai.dto';

@Injectable()
export class ClaudeService {
  private readonly CLAUDE_API_KEY: string;
  private readonly CLAUDE_MODEL: string;
  private readonly ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';

  constructor(private readonly configService: ConfigService) {
    this.CLAUDE_API_KEY = this.configService.get('CLAUDE_API_KEY') || this.configService.get('ANTHROPIC_API_KEY') || '';
    this.CLAUDE_MODEL = this.configService.get('CLAUDE_MODEL') || 'claude-sonnet-4-6';
  }

  get isConfigured(): boolean {
    return !!this.CLAUDE_API_KEY;
  }

  private get anthropicHeaders() {
    return {
      'x-api-key': this.CLAUDE_API_KEY,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    };
  }

  /**
   * Translates OpenAI-formatted messages to Anthropic messages.
   * Maps system messages to a top-level system parameter and aggregates consecutive tool responses
   * into single user messages with multiple tool results.
   */
  translateMessages(openaiMessages: ChatCompletionRequestMessage[]): { systemPrompt: string; messages: any[] } {
    const anthropicMessages: any[] = [];
    let systemPrompt = '';

    for (const msg of openaiMessages) {
      if (msg.role === 'system') {
        systemPrompt = systemPrompt ? `${systemPrompt}\n\n${msg.content}` : msg.content;
        continue;
      }

      if (msg.role === 'tool') {
        const toolResultBlock = {
          type: 'tool_result',
          tool_use_id: (msg as any).tool_call_id || '',
          content: msg.content,
        };

        const lastMsg = anthropicMessages[anthropicMessages.length - 1];
        if (lastMsg && lastMsg.role === 'user' && Array.isArray(lastMsg.content)) {
          lastMsg.content.push(toolResultBlock);
        } else {
          anthropicMessages.push({
            role: 'user',
            content: [toolResultBlock],
          });
        }
        continue;
      }

      if (msg.role === 'assistant') {
        const toolCalls = (msg as any).tool_calls;
        if (toolCalls && toolCalls.length > 0) {
          const contentBlocks: any[] = [];
          if (msg.content) {
            contentBlocks.push({ type: 'text', text: msg.content });
          }
          for (const call of toolCalls) {
            let input = {};
            try {
              input = JSON.parse(call.function?.arguments || '{}');
            } catch {
              input = {};
            }
            contentBlocks.push({
              type: 'tool_use',
              id: call.id,
              name: call.function?.name,
              input,
            });
          }
          anthropicMessages.push({
            role: 'assistant',
            content: contentBlocks,
          });
        } else {
          anthropicMessages.push({
            role: 'assistant',
            content: msg.content,
          });
        }
        continue;
      }

      if (msg.role === 'user') {
        anthropicMessages.push({
          role: 'user',
          content: msg.content,
        });
      }
    }

    return { systemPrompt, messages: anthropicMessages };
  }

  /**
   * Translates OpenAI tool schemas into Anthropic's tool schema format.
   */
  translateTools(openaiTools: any[]): any[] | undefined {
    if (!openaiTools) return undefined;
    return openaiTools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));
  }

  /**
   * Streams chat completions from Claude's SSE endpoint to the client.
   *
   * Returns `true` once it has taken ownership of the response (headers flushed
   * and streaming/ended). Returns `false` if the upstream call fails BEFORE any
   * bytes are sent — in that case `res` is untouched, so the caller can fall back
   * to another provider. Headers are only flushed after the upstream connection
   * is established, which is what makes a clean fallback possible.
   */
  async streamClaude(messages: ChatCompletionRequestMessage[], res: Response): Promise<boolean> {
    const { systemPrompt, messages: anthropicMessages } = this.translateMessages(messages);

    let response: any;
    try {
      response = await axios.post(
        this.ANTHROPIC_URL,
        {
          model: this.CLAUDE_MODEL,
          messages: anthropicMessages,
          system: systemPrompt || undefined,
          max_tokens: 4096,
          temperature: 0.2,
          stream: true,
        },
        {
          responseType: 'stream',
          headers: this.anthropicHeaders,
        },
      );
    } catch (err: any) {
      // Upstream rejected the request (bad key, retired model, etc.) before we
      // sent anything to the client — log it and let the caller fall back.
      await this.logStreamStartError(err);
      return false;
    }

    // Connection established: now it's safe to commit to streaming this response.
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    return new Promise<boolean>((resolve) => {
      let buffer = '';
      response.data.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payloadStr = trimmed.replace(/^data:\s*/, '');
          try {
            const data = JSON.parse(payloadStr);
            if (data.type === 'content_block_delta' && data.delta?.text) {
              res.write(data.delta.text);
              (res as any).flush?.();
            }
          } catch {
            // ignore partial chunks
          }
        }
      });

      response.data.on('end', () => {
        if (!res.writableEnded) res.end();
        resolve(true);
      });

      // Mid-stream error: headers are already sent, so we can't fall back —
      // just close out the (partial) response.
      response.data.on('error', (err: any) => {
        console.error('Claude stream error:', err);
        if (!res.writableEnded) res.end();
        resolve(true);
      });
    });
  }

  /** Decodes and logs an Anthropic error response (which may be a gzipped stream). */
  private async logStreamStartError(err: any): Promise<void> {
    if (!(err?.response?.data && typeof err.response.data.on === 'function')) {
      console.error('Failed to start Claude stream:', err?.response?.data || err?.message);
      return;
    }
    const stream = err.response.data;
    const encoding = (err.response.headers || {})['content-encoding'] || '';
    let decoderStream = stream;
    try {
      if (encoding.includes('gzip')) {
        decoderStream = stream.pipe(require('zlib').createGunzip());
      } else if (encoding.includes('deflate')) {
        decoderStream = stream.pipe(require('zlib').createInflate());
      }
    } catch (pipeErr: any) {
      console.error('Error decompressing Claude error stream:', pipeErr.message);
    }
    await new Promise<void>((resolve) => {
      let body = '';
      decoderStream.on('data', (chunk: Buffer) => (body += chunk.toString()));
      decoderStream.on('end', () => {
        console.error('Failed to start Claude stream. Response:', body);
        resolve();
      });
      decoderStream.on('error', (e: any) => {
        console.error('Failed to start Claude stream. Read error:', e.message);
        resolve();
      });
    });
  }

  /**
   * Generates a non-streaming response from Claude (used for classification tasks).
   */
  async claudeGenerate(prompt: string, temperature = 0): Promise<string> {
    try {
      const res = await axios.post(
        this.ANTHROPIC_URL,
        {
          model: this.CLAUDE_MODEL,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 2048,
          temperature,
        },
        {
          headers: this.anthropicHeaders,
        },
      );
      return res.data?.content?.[0]?.text || '';
    } catch (error: any) {
      console.error('Claude generate error:', error?.response?.data || error?.message);
      return '';
    }
  }

  /**
   * Executes a single turn of reasoning / tool selection, mapping the response back to OpenAI's tool format.
   */
  async claudeToolTurn(messages: ChatCompletionRequestMessage[], tools: any[]): Promise<any> {
    const { systemPrompt, messages: anthropicMessages } = this.translateMessages(messages);
    const anthropicTools = this.translateTools(tools);

    try {
      const res = await axios.post(
        this.ANTHROPIC_URL,
        {
          model: this.CLAUDE_MODEL,
          messages: anthropicMessages,
          system: systemPrompt || undefined,
          tools: anthropicTools,
          max_tokens: 4096,
          temperature: 0.2,
        },
        {
          headers: this.anthropicHeaders,
        },
      );

      const content = res.data?.content || [];
      let textContent = '';
      const toolCalls: any[] = [];

      for (const block of content) {
        if (block.type === 'text') {
          textContent = textContent ? `${textContent}\n${block.text}` : block.text;
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            type: 'function',
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input),
            },
          });
        }
      }

      return {
        role: 'assistant',
        content: textContent || null,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      };
    } catch (error: any) {
      console.error('Claude tool turn error:', error?.response?.data || error?.message);
      throw new InternalServerErrorException(`Claude tool turn failed: ${error?.message}`);
    }
  }
}
