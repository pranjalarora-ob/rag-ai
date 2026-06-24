import { Injectable } from '@nestjs/common';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { ClaudeService } from './claude.service';
import { RerankService } from './rerank.service';
import { ProjectQueryService } from './project-query.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

const MAX_STEPS = 4;

export interface AgentResult {
  answer: string;
  trace: Array<{ tool: string; args: any }>;
  steps: number;
}

/**
 * The AGENT: turns a natural-language question into a STRUCTURED query and runs it
 * deterministically. The LLM (Claude if configured, else OpenRouter) only fills the
 * `queryProjects` tool's arguments — all filtering/sorting/counting is done in code,
 * so results are exact (no truncation, no invented numbers). `searchProjects` handles
 * descriptive/semantic lookups about a specific project.
 */
@Injectable()
export class ProjectAgentService {
  constructor(
    private readonly openai: OpenaiService,
    private readonly qdrant: QdrantService,
    private readonly claude: ClaudeService,
    private readonly rerank: RerankService,
    private readonly projectQuery: ProjectQueryService,
  ) {}

  private readonly tools = [
    {
      type: 'function',
      function: {
        name: 'queryProjects',
        description:
          'Filter, sort, count, or list the customer\'s projects/leads with EXACT results. ' +
          'Use for ANY question that filters by city, owner, zone, area, estimated value, project code, ' +
          'or type (lead vs project), and for counts, "top N", or listing. ' +
          'Convert worded amounts to plain numbers: "40 lakhs" -> 4000000, "2 cr"/"2 crore" -> 20000000, "5k" -> 5000. ' +
          'For "top N largest area" set sortBy=areaSft,sortDir=desc,limit=N; for "top N by value" sortBy=estimatedValue,sortDir=desc,limit=N. ' +
          'Use operation="count" when the user asks how many / count.',
        parameters: {
          type: 'object',
          properties: {
            type: { type: 'string', enum: ['lead', 'project'], description: 'Filter to leads or projects; omit for both.' },
            city: { type: 'string', description: 'Exact city name, e.g. "Gurugram".' },
            owner: { type: 'string', description: 'Owner/team name, e.g. "SMB Team" (typo-tolerant).' },
            zone: { type: 'string', description: 'Zone name, e.g. "Gurgaon".' },
            projectCode: { type: 'number', description: 'Exact project code.' },
            area: {
              type: 'object',
              description: 'Area condition in sqft.',
              properties: {
                op: { type: 'string', enum: ['gt', 'gte', 'lt', 'lte', 'eq'] },
                value: { type: 'number' },
              },
            },
            estimatedValue: {
              type: 'object',
              description: 'Estimated value condition in rupees (convert lakh/crore to a plain number).',
              properties: {
                op: { type: 'string', enum: ['gt', 'gte', 'lt', 'lte', 'eq'] },
                value: { type: 'number' },
              },
            },
            sortBy: {
              type: 'string',
              enum: ['areaSft', 'estimatedValue', 'currentProjectValue', 'closureValue', 'projectCode'],
            },
            sortDir: { type: 'string', enum: ['asc', 'desc'] },
            limit: { type: 'number', description: 'Max rows (e.g. "top 5" -> 5).' },
            operation: { type: 'string', enum: ['list', 'count'], description: 'count = number only; list = rows.' },
          },
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'searchProjects',
        description:
          'Semantic search for DESCRIPTIVE questions about a specific project (notes, who is on the team, ' +
          'context, "tell me about ..."). Returns matching project summaries. Not for filtering/counting.',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string', description: 'Natural-language search query.' } },
          required: ['query'],
        },
      },
    },
  ];

  async run(params: { question: string; customerId: string }): Promise<AgentResult> {
    const { question, customerId } = params;

    const system = `${SYSTEM_PROMPT}

You are a project assistant. Use a tool to answer — never invent numbers or lists.
- The customerId is "${customerId}" and is already known; never ask for it.
- For filtering / counting / "top N" / listing by city, owner, zone, area, value, or lead-vs-project: call queryProjects.
- If the question names a specific project CODE (a long number like 2022072073) — even "tell me about <code>" — call queryProjects with projectCode. searchProjects CANNOT find exact codes.
- For descriptive questions with NO code (notes, who is on the team, "projects about call centres"): call searchProjects.
- Convert worded amounts to numbers (40 lakhs -> 4000000, 2 cr -> 20000000).`;

    const messages: any[] = [
      { role: 'system', content: system },
      { role: 'user', content: question },
    ];
    const trace: AgentResult['trace'] = [];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const assistant = this.claude.isConfigured
        ? await this.claude.claudeToolTurn(messages, this.tools)
        : await this.openai.openRouterToolTurn(messages, this.tools);

      const toolCalls = assistant?.tool_calls;
      if (!toolCalls || toolCalls.length === 0) {
        return { answer: assistant?.content ?? '', trace, steps: step };
      }

      const call = toolCalls[0];
      let args: any = {};
      try {
        args = JSON.parse(call.function?.arguments || '{}');
      } catch {
        args = {};
      }
      trace.push({ tool: call.function?.name, args });

      // queryProjects is deterministic — return its exact result directly, bypassing
      // the LLM so the table/count is never truncated or reordered.
      if (call.function?.name === 'queryProjects') {
        const result = await this.projectQuery.execute(args, customerId);
        return { answer: result.answer, trace, steps: step };
      }

      // searchProjects feeds results back so the LLM can phrase a descriptive answer.
      if (call.function?.name === 'searchProjects') {
        const reranked = await this.searchProjects(args.query || '', customerId);
        messages.push(assistant);
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(reranked) });
        continue;
      }

      return { answer: `Unknown tool: ${call.function?.name}`, trace, steps: step };
    }

    return { answer: 'Could not complete the request within the step limit.', trace, steps: MAX_STEPS };
  }

  private async searchProjects(query: string, customerId: string) {
    const embedding = await this.openai.generateEmbedding(query);
    const hits = await this.qdrant.search(COLLECTION, {
      vector: embedding,
      limit: 50,
      filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
    });
    const rawChunks = hits.sort((a: any, b: any) => b.score - a.score).map((h: any) => h.payload?.text as string);
    return (await this.rerank.rerank(query, rawChunks, 10)).slice(0, 10);
  }
}
