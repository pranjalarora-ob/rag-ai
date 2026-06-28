import { Injectable } from '@nestjs/common';
import { Response } from 'express';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { RerankService } from './rerank.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

const MAX_STEPS = 5; // safety cap on the tool→evaluate→tool loop

// Appended to the planner's system prompt so list-style answers render as tables
// the UI can turn into charts. The chart parser needs a "Name" column plus a
// "Value"/"Area" column, so the columns below must stay in sync with it.
const TABLE_FORMAT_INSTRUCTION = `
When the answer lists multiple projects/records, format them as a GitHub markdown table with these exact columns:
| # | Code | Name | City | Area (sqft) | Estimated Value |
Use one row per record and keep raw numbers (no "Cr"/"L" suffixes) in the value/area cells so they can be charted. For single facts, counts, or short explanations, answer in plain prose instead.`;

export interface PlannerResult {
  answer: string;
  trace: Array<{ tool: string; args: any; result: any }>;
  steps: number;
}

/**
 * The PLANNER ("thinker"): an LLM tool-calling loop. It decides which tool(s) to call
 * (searchProjects = RAG lookup, projectAnalytics = exact math), runs them, reads the
 * results, and drafts the final answer. Generation via OpenRouter (OpenAI tool format).
 */
@Injectable()
export class PlannerService {
  constructor(
    private readonly openaiService: OpenaiService,
    private readonly qdrantService: QdrantService,
    private readonly projectAnalyticsService: ProjectAnalyticsService,
    private readonly rerankService: RerankService,
  ) { }

  private readonly tools = [
    {
      type: 'function',
      function: {
        name: 'searchProjects',
        description:
          'Search project records by meaning. Use for DESCRIPTIVE lookups about a specific project — who is the design manager, what stage, details/notes. Returns matching project summaries.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'A natural-language search query.' },
          },
          required: ['query'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'projectAnalytics',
        description:
          'Compute EXACT aggregates over ALL of the customer\'s projects: totals, averages, counts, top-N rankings, filtered sums/lists. Use for any "total", "how many", "top N", "highest/lowest", "average", or "where <person> is <role>" question. Never compute numbers yourself — call this.',
        parameters: {
          type: 'object',
          properties: {
            metric: { type: 'string', enum: ['estimatedValue', 'boqValue'] },
            topN: { type: 'number' },
            teamMember: { type: 'string', description: 'Filter to projects where this person is on the team.' },
            role: { type: 'string', description: 'Role to match with teamMember, e.g. "Design Manager".' },
            groupBy: { type: 'string', enum: ['stage', 'city', 'owner'] },
            city: { type: 'string', description: 'Filter to projects in this city, e.g., "Gurugram".' },
            zone: { type: 'string', description: 'Filter to projects in this zone, e.g., "North".' },
            stage: { type: 'string', description: 'Filter to projects in this stage.' },
            owner: { type: 'string', description: 'Filter to projects owned by this person/team.' },
            minValue: { type: 'number', description: 'Filter to projects with estimatedValue >= this number (in rupees).' },
            maxValue: { type: 'number', description: 'Filter to projects with estimatedValue <= this number (in rupees).' },
          },
        },
      },
    },
  ];

  private buildSystemInstruction(customerId: string): string {
    return `${SYSTEM_PROMPT}

You are the PLANNER. Decide which tool(s) to call to answer the user, call them, read the results, then write the final answer.
Rules:
- The customerId is "${customerId}" and is already known — never ask the user for it.
- For totals, counts, averages, rankings, or filtered sums/lists, you MUST call projectAnalytics and use its exact numbers. Do not calculate numbers yourself.
- For descriptive questions about a specific project, call searchProjects.
- Base every fact and number ONLY on tool results. If the tools return nothing relevant, say you don't have that information.
- When you have enough information, reply with the final answer as plain text (no tool call).
${TABLE_FORMAT_INSTRUCTION}`;
  }

  // Run the tool-calling loop until the model has everything it needs to answer.
  // Returns { messages, trace, done }: `done` is the final answer text if the model
  // produced one without needing a final streamed turn, else null. `messages` holds
  // the full conversation (incl. tool results) ready for a final answer turn.
  private async resolveTools(
    messages: any[],
    customerId: string,
  ): Promise<{ trace: PlannerResult['trace']; finalText: string | null; steps: number }> {
    const trace: PlannerResult['trace'] = [];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const assistant = await this.openaiService.openRouterToolTurn(messages, this.tools);
      const toolCalls = assistant?.tool_calls;

      if (!toolCalls || toolCalls.length === 0) {
        return { trace, finalText: assistant?.content ?? '', steps: step };
      }

      messages.push(assistant);

      for (const call of toolCalls) {
        let args: any = {};
        try {
          args = JSON.parse(call.function?.arguments || '{}');
        } catch {
          args = {};
        }
        const result = await this.executeTool(call.function?.name, args, customerId);
        trace.push({ tool: call.function?.name, args, result });
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
      }
    }

    return { trace, finalText: null, steps: MAX_STEPS };
  }

  async run(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
  }): Promise<PlannerResult> {
    const { question, customerId, history = [] } = params;

    const messages: any[] = [
      { role: 'system', content: this.buildSystemInstruction(customerId) },
      ...history,
      { role: 'user', content: question },
    ];

    const { trace, finalText, steps } = await this.resolveTools(messages, customerId);
    return {
      answer: finalText ?? 'I could not complete the request within the allowed number of steps.',
      trace,
      steps,
    };
  }

  // Streaming variant: resolves tools the same way, then STREAMS the final answer
  // token-by-token to the HTTP response (for the chat UI's typing effect).
  async runStream(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
    res: Response;
  }): Promise<void> {
    const { question, customerId, history = [], res } = params;

    const messages: any[] = [
      { role: 'system', content: this.buildSystemInstruction(customerId) },
      ...history,
      { role: 'user', content: question },
    ];

    // Drive the tool loop. Its final turn already produces the answer text, so we
    // reuse that instead of re-generating it with a second streaming LLM call —
    // saving one full generation round-trip per request.
    const { finalText } = await this.resolveTools(messages, customerId);
    const answer = finalText || 'I could not complete the request within the allowed number of steps.';

    // Emulate token streaming by writing the answer in small chunks so the UI keeps
    // its typing effect, without the cost of a real second streamed generation.
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.flushHeaders?.();

    const tokens = answer.split(/(\s+)/); // keep whitespace tokens
    for (let i = 0; i < tokens.length; i += 3) {
      if (res.writableEnded) break;
      res.write(tokens.slice(i, i + 3).join(''));
      (res as any).flush?.();
      await new Promise((r) => setTimeout(r, 6));
    }
    if (!res.writableEnded) res.end();
  }

  private async executeTool(name: string, args: any, customerId: string) {
    try {
      if (name === 'searchProjects') {
        const embedding = await this.openaiService.generateEmbedding(args.query || '');
        const hits = await this.qdrantService.search(COLLECTION, {
          vector: embedding,
          limit: 50,
          filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
        });
        const rawChunks = hits
          .sort((a: any, b: any) => b.score - a.score)
          .map((h: any) => h.payload?.text as string);
        const reranked = await this.rerankService.rerank(args.query || '', rawChunks, 10);
        return reranked.slice(0, 10);
      }

      if (name === 'projectAnalytics') {
        return this.projectAnalyticsService.analyze({
          customerId,
          metric: args.metric,
          topN: args.topN,
          teamMember: args.teamMember,
          role: args.role,
          groupBy: args.groupBy,
          city: args.city,
          zone: args.zone,
          stage: args.stage,
          owner: args.owner,
          minValue: args.minValue,
          maxValue: args.maxValue,
        });
      }

      return { error: `Unknown tool: ${name}` };
    } catch (err: any) {
      return { error: err?.message || 'tool execution failed' };
    }
  }
}
