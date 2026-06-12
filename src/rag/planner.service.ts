import { Injectable } from '@nestjs/common';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

const MAX_STEPS = 5; // safety cap on the tool→evaluate→tool loop

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
  ) {}

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
          },
        },
      },
    },
  ];

  async run(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
  }): Promise<PlannerResult> {
    const { question, customerId, history = [] } = params;

    const systemInstruction = `${SYSTEM_PROMPT}

You are the PLANNER. Decide which tool(s) to call to answer the user, call them, read the results, then write the final answer.
Rules:
- The customerId is "${customerId}" and is already known — never ask the user for it.
- For totals, counts, averages, rankings, or filtered sums/lists, you MUST call projectAnalytics and use its exact numbers. Do not calculate numbers yourself.
- For descriptive questions about a specific project, call searchProjects.
- Base every fact and number ONLY on tool results. If the tools return nothing relevant, say you don't have that information.
- When you have enough information, reply with the final answer as plain text (no tool call).`;

    const messages: any[] = [
      { role: 'system', content: systemInstruction },
      ...history,
      { role: 'user', content: question },
    ];

    const trace: PlannerResult['trace'] = [];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const assistant = await this.openaiService.openRouterToolTurn(messages, this.tools);
      const toolCalls = assistant?.tool_calls;

      if (!toolCalls || toolCalls.length === 0) {
        return { answer: assistant?.content ?? '', trace, steps: step };
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

    return { answer: 'I could not complete the request within the allowed number of steps.', trace, steps: MAX_STEPS };
  }

  private async executeTool(name: string, args: any, customerId: string) {
    try {
      if (name === 'searchProjects') {
        const embedding = await this.openaiService.generateGeminiEmbedding(args.query || '');
        const hits = await this.qdrantService.search(COLLECTION, {
          vector: embedding,
          limit: 20,
          filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
        });
        return hits
          .sort((a: any, b: any) => b.score - a.score)
          .slice(0, 10)
          .map((h: any) => h.payload?.text as string);
      }

      if (name === 'projectAnalytics') {
        return this.projectAnalyticsService.analyze({
          customerId,
          metric: args.metric,
          topN: args.topN,
          teamMember: args.teamMember,
          role: args.role,
          groupBy: args.groupBy,
        });
      }

      return { error: `Unknown tool: ${name}` };
    } catch (err: any) {
      return { error: err?.message || 'tool execution failed' };
    }
  }
}
