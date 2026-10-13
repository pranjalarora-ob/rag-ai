import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Anthropic from '@anthropic-ai/sdk';
import { Response } from 'express';
import { ChatCompletionRequestMessage } from './dto/openai.dto';

export type Effort = 'low' | 'medium' | 'high';

// Room for adaptive thinking plus a long markdown table answer.
const MAX_TOKENS = 8192;

// The raw Claude content of an assistant turn rides on the OpenAI-shaped message so
// the next tool-loop step replays it unmodified (thinking blocks must be passed back
// as-is). Non-enumerable, so it never leaks into an OpenRouter request on fallback.
const RAW_CONTENT = Symbol('claudeRawContent');

/**
 * The LLM for all chat work (routing, planning, answer writing) — Claude via the
 * official Anthropic SDK. Callers keep using the OpenAI message / tool format used
 * across the codebase; this service translates both ways. Sampling params
 * (temperature) are never sent — Claude 5-family models reject them.
 * Embeddings are separate (OpenaiService → OpenRouter).
 */
@Injectable()
export class ClaudeService {
  private readonly logger = new Logger(ClaudeService.name);
  private readonly client: Anthropic | null;
  private readonly model: string;
  private readonly effort: Effort;

  constructor(private readonly configService: ConfigService) {
    const apiKey = this.configService.get('CLAUDE_API_KEY') || this.configService.get('ANTHROPIC_API_KEY') || '';
    this.client = apiKey ? new Anthropic({ apiKey, timeout: 120_000, maxRetries: 2 }) : null;
    this.model = this.configService.get('CLAUDE_MODEL') || 'claude-haiku-5-5';
    this.effort = (this.configService.get('CLAUDE_EFFORT') as Effort) || 'low';
    if (this.client) this.logger.log(`LLM: ${this.model}`);
    else this.logger.warn('ANTHROPIC_API_KEY is not set — chat/LLM calls will fail');
  }

  get isConfigured(): boolean {
    return !!this.client;
  }

  get modelName(): string {
    return this.model;
  }

  // `effort` is only accepted by newer models (e.g. Haiku 4.5 rejects it).
  private outputConfig(effort?: Effort) {
    const supported = /^claude-(haiku-5|sonnet-5|opus-5|fable|mythos|opus-4-[6-8]|sonnet-4-6)/.test(this.model);
    return supported ? { output_config: { effort: effort ?? this.effort } } : {};
  }

  /**
   * Translates OpenAI-formatted messages to Claude messages. System messages become
   * top-level system blocks. With `cacheSystem`, the first one (a stable instruction
   * prompt) is marked for prompt caching so repeat requests pay ~10% for it and the
   * tools before it — leave it off when the system prompt carries per-request context.
   */
  translateMessages(
    openaiMessages: ChatCompletionRequestMessage[],
    cacheSystem = false,
  ): {
    system: Anthropic.TextBlockParam[];
    messages: Anthropic.MessageParam[];
  } {
    const systemParts: string[] = [];
    const messages: Anthropic.MessageParam[] = [];

    for (const msg of openaiMessages as any[]) {
      if (msg.role === 'system') {
        if (msg.content) systemParts.push(msg.content);
        continue;
      }

      if (msg.role === 'tool') {
        const block: Anthropic.ToolResultBlockParam = {
          type: 'tool_result',
          tool_use_id: msg.tool_call_id || '',
          content: String(msg.content ?? ''),
        };
        const last = messages[messages.length - 1];
        if (last?.role === 'user' && Array.isArray(last.content)) {
          last.content.push(block);
        } else {
          messages.push({ role: 'user', content: [block] });
        }
        continue;
      }

      if (msg.role === 'assistant') {
        if (msg[RAW_CONTENT]) {
          messages.push({ role: 'assistant', content: msg[RAW_CONTENT] });
          continue;
        }
        const toolCalls = msg.tool_calls;
        if (toolCalls?.length) {
          const content: Anthropic.ContentBlockParam[] = [];
          if (msg.content) content.push({ type: 'text', text: msg.content });
          for (const call of toolCalls) {
            let input = {};
            try {
              input = JSON.parse(call.function?.arguments || '{}');
            } catch {
              input = {};
            }
            content.push({ type: 'tool_use', id: call.id, name: call.function?.name, input });
          }
          messages.push({ role: 'assistant', content });
        } else if (msg.content) {
          messages.push({ role: 'assistant', content: msg.content });
        }
        continue;
      }

      if (msg.role === 'user' && msg.content) {
        messages.push({ role: 'user', content: msg.content });
      }
    }

    // A conversation must open with a user turn — drop leading assistant history.
    while (messages.length && messages[0].role !== 'user') messages.shift();
    this.fillMissingToolResults(messages);

    const system: Anthropic.TextBlockParam[] = systemParts.map((text, i) =>
      cacheSystem && i === 0 ? { type: 'text', text, cache_control: { type: 'ephemeral' } } : { type: 'text', text },
    );
    return { system, messages };
  }

  // Claude requires a tool_result for every tool_use. Callers that act only on the
  // first call (project-agent) would otherwise get a 400 after a parallel tool call.
  private fillMissingToolResults(messages: Anthropic.MessageParam[]) {
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;
      const ids = msg.content.filter((b) => b.type === 'tool_use').map((b: any) => b.id as string);
      if (!ids.length) continue;

      let next = messages[i + 1];
      if (!next || next.role !== 'user' || !Array.isArray(next.content)) {
        next = { role: 'user', content: [] };
        messages.splice(i + 1, 0, next);
      }
      const content = next.content as Anthropic.ContentBlockParam[];
      const answered = new Set(content.filter((b) => b.type === 'tool_result').map((b: any) => b.tool_use_id));
      for (const id of ids) {
        if (!answered.has(id)) content.push({ type: 'tool_result', tool_use_id: id, content: 'Not executed.' });
      }
    }
  }

  /** Translates OpenAI tool schemas into Claude's tool schema format. */
  translateTools(openaiTools: any[]): Anthropic.Tool[] | undefined {
    if (!openaiTools) return undefined;
    return openaiTools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));
  }

  /**
   * Streams the answer to the client as plain text and returns the full text.
   * Returns `null` if the call fails BEFORE any bytes are sent (res untouched, so the
   * caller can still send an error status).
   */
  async stream(messages: ChatCompletionRequestMessage[], res: Response): Promise<string | null> {
    if (!this.client) return null;
    const { system, messages: claudeMessages } = this.translateMessages(messages);

    const start = () =>
      this.client.messages.create({
        model: this.model,
        max_tokens: MAX_TOKENS,
        system,
        messages: claudeMessages,
        stream: true,
        ...this.outputConfig(),
      });
    let events: Awaited<ReturnType<typeof start>>;
    try {
      events = await start();
    } catch (error) {
      this.logError('stream start', error);
      return null;
    }

    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    // Stop generating (and paying for) tokens if the client disconnects.
    res.on('close', () => events.controller.abort());

    // Mid-stream errors can't change the status (headers are sent) — close the partial response.
    let text = '';
    try {
      for await (const event of events) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          text += event.delta.text;
          res.write(event.delta.text);
          (res as any).flush?.();
        }
      }
    } catch (error) {
      if (!res.writableEnded) this.logError('stream', error);
    }
    if (!res.writableEnded) res.end();
    return text;
  }

  /** One-shot prompt → text (routing/classification, answer phrasing). Returns '' on failure. */
  async generate(prompt: string, opts: { effort?: Effort } = {}): Promise<string> {
    return this.chat([{ role: 'user', content: prompt }], opts);
  }

  /** OpenAI-format conversation → text. Returns '' on failure. */
  async chat(messages: ChatCompletionRequestMessage[], opts: { effort?: Effort } = {}): Promise<string> {
    if (!this.client) return '';
    const { system, messages: claudeMessages } = this.translateMessages(messages);
    try {
      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: MAX_TOKENS,
        system,
        messages: claudeMessages,
        ...this.outputConfig(opts.effort),
      });
      this.logUsage('chat', response);
      return this.textOf(response);
    } catch (error) {
      this.logError('chat', error);
      return '';
    }
  }

  /**
   * One turn of tool selection. Returns an OpenAI-shaped assistant message (content +
   * tool_calls) carrying the raw Claude content for the next step. Throws on failure.
   */
  async toolTurn(messages: ChatCompletionRequestMessage[], tools: any[]): Promise<any> {
    if (!this.client) throw new Error('ANTHROPIC_API_KEY is not set');
    const { system, messages: claudeMessages } = this.translateMessages(messages, true);
    try {
      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: MAX_TOKENS,
        system,
        tools: this.translateTools(tools),
        messages: claudeMessages,
        ...this.outputConfig(),
      });
      this.logUsage('toolTurn', response);

      const toolCalls = response.content
        .filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use')
        .map((b) => ({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input) } }));
      const assistant: any = {
        role: 'assistant',
        content: this.textOf(response) || null,
        tool_calls: toolCalls.length ? toolCalls : undefined,
      };
      Object.defineProperty(assistant, RAW_CONTENT, { value: response.content, enumerable: false });
      return assistant;
    } catch (error) {
      this.logError('toolTurn', error);
      throw error;
    }
  }

  // Responses can open with thinking blocks — read text by block type, not position.
  private textOf(response: Anthropic.Message): string {
    return response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('\n')
      .trim();
  }

  private logUsage(kind: string, response: Anthropic.Message) {
    const u = response.usage;
    this.logger.debug(
      `${kind} ${response.model} stop=${response.stop_reason} in=${u.input_tokens} ` +
        `cacheRead=${u.cache_read_input_tokens ?? 0} cacheWrite=${u.cache_creation_input_tokens ?? 0} out=${u.output_tokens}`,
    );
    if (response.stop_reason === 'max_tokens') this.logger.warn(`${kind}: hit max_tokens — answer may be cut off`);
    if (response.stop_reason === 'refusal') this.logger.warn(`${kind}: model declined the request`);
  }

  private logError(kind: string, error: unknown) {
    if (error instanceof Anthropic.APIError) {
      this.logger.error(`Claude ${kind} failed: ${error.status} ${error.message}`);
    } else {
      this.logger.error(`Claude ${kind} failed: ${(error as any)?.message ?? error}`);
    }
  }
}
