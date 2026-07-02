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
  ) { }

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
          'context, "tell me about ..."). Returns matching project summaries. Not for filtering/counting. Let the LLM decide which filters should apply.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'A natural-language semantic search query.' },
            projectCode: { type: 'integer', description: 'Filter by exact project code.' },
            city: { type: 'string', description: 'Filter by exact city name.' },
            status: { type: 'string', description: 'Filter by project status, e.g. "Cancelled", "InProgress".' },
            type: { type: 'string', enum: ['lead', 'project'], description: 'Filter to leads or projects.' },
            docType: { type: 'string', enum: ['project', 'project-flow-phase', 'boq'], description: 'Filter to specific document type.' },
            areaMin: { type: 'number', description: 'Min area in sqft.' },
            areaMax: { type: 'number', description: 'Max area in sqft.' },
            valueMin: { type: 'number', description: 'Min estimated value in rupees.' },
            valueMax: { type: 'number', description: 'Max estimated value in rupees.' },
            accountId: { type: 'string', description: 'Filter by account ID.' },
            projectId: { type: 'string', description: 'Filter by project ID.' },
            teamMemberId: { type: 'string', description: 'Filter by team member user ID.' },
            parentId: { type: 'string', description: 'Filter by parent stage/milestone stage ID.' },
          },
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
        const reranked = await this.searchProjects(args, customerId, question);
        messages.push(assistant);
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(reranked) });
        continue;
      }

      return { answer: `Unknown tool: ${call.function?.name}`, trace, steps: step };
    }

    return { answer: 'Could not complete the request within the step limit.', trace, steps: MAX_STEPS };
  }

  private async searchProjects(args: any, customerId: string, originalQuestion?: string) {
    const query = args.query || '';
    const embedding = await this.openai.generateEmbedding(query);

    const filterMust: any[] = [{ key: 'customerId', match: { value: customerId } }];

    // 1. projectCode
    const projectCode = args.projectCode || query.match(/\b(\d{6,})\b/)?.[1] || (originalQuestion || '').match(/\b(\d{6,})\b/)?.[1];
    if (projectCode) {
      filterMust.push({
        key: 'projectCode',
        match: { value: String(projectCode) }
      });
    }

    // 2. city
    let city = args.city;
    if (!city && !projectCode) {
      city = this.parseCity(query);
      if (!city && originalQuestion) {
        city = this.parseCity(originalQuestion);
      }
    }
    if (city) {
      const capitalized = city.charAt(0).toUpperCase() + city.slice(1).toLowerCase();
      filterMust.push({
        key: 'city',
        match: { any: [city, city.toLowerCase(), city.toUpperCase(), capitalized] }
      });
    }

    // 3. status
    if (args.status) {
      filterMust.push({ key: 'status', match: { value: args.status } });
    }

    // 4. type
    let typeFilter = args.type;
    if (!typeFilter && !projectCode) {
      typeFilter = this.parseType(query);
      if (!typeFilter && originalQuestion) {
        typeFilter = this.parseType(originalQuestion);
      }
    }
    if (typeFilter) {
      filterMust.push({ key: 'type', match: { value: typeFilter } });
    }

    // 5. docType
    if (args.docType) {
      filterMust.push({ key: 'docType', match: { value: args.docType } });
    } else {
      const isFlowPhaseQuery = /phase|milestone|stage|workflow/i.test(query) || 
        (originalQuestion && /phase|milestone|stage|workflow/i.test(originalQuestion));
      if (!isFlowPhaseQuery && !projectCode) {
        filterMust.push({ key: 'docType', match: { value: 'project' } });
      }
    }

    // 6. accountId
    if (args.accountId) {
      filterMust.push({ key: 'accountId', match: { value: args.accountId } });
    }

    // 7. projectId
    if (args.projectId) {
      filterMust.push({ key: 'projectId', match: { value: args.projectId } });
    }

    // 8. teamMemberId
    if (args.teamMemberId) {
      filterMust.push({ key: 'team[].userId', match: { value: args.teamMemberId } });
    }

    // 9. parentId
    if (args.parentId) {
      filterMust.push({ key: 'parentId', match: { value: args.parentId } });
    }

    // 10. area range
    if (args.areaMin !== undefined || args.areaMax !== undefined) {
      const range: any = {};
      if (args.areaMin !== undefined) range.gte = args.areaMin;
      if (args.areaMax !== undefined) range.lte = args.areaMax;
      filterMust.push({ key: 'areaSft', range });
    } else if (!projectCode) {
      let areaRange = this.parseNumericRange(query, ['area', 'sqft', 'sft', 'square feet', 'square foot']);
      if (!areaRange && originalQuestion) {
        areaRange = this.parseNumericRange(originalQuestion, ['area', 'sqft', 'sft', 'square feet', 'square foot']);
      }
      if (areaRange) {
        filterMust.push({ key: 'areaSft', range: areaRange });
      }
    }

    // 11. estimatedValue range
    if (args.valueMin !== undefined || args.valueMax !== undefined) {
      const range: any = {};
      if (args.valueMin !== undefined) range.gte = args.valueMin;
      if (args.valueMax !== undefined) range.lte = args.valueMax;
      filterMust.push({ key: 'estimatedValue', range });
    } else if (!projectCode) {
      let valueRange = this.parseNumericRange(query, ['estimated value', 'estimatedvalue', 'value', 'budget', 'worth']);
      if (!valueRange && originalQuestion) {
        valueRange = this.parseNumericRange(originalQuestion, ['estimated value', 'estimatedvalue', 'value', 'budget', 'worth']);
      }
      if (valueRange) {
        filterMust.push({ key: 'estimatedValue', range: valueRange });
      }
    }
    console.log("filterMust", filterMust);
    let hits: any[];
    if (projectCode) {
      hits = await this.qdrant.scrollAll(COLLECTION, { must: filterMust });
    } else {
      hits = await this.qdrant.search(COLLECTION, {
        vector: embedding,
        limit: 10,
        filter: { must: filterMust },
      });
    }
    console.log('................', hits)
    console.log(`[Qdrant Search - Agent] Received ${hits?.length || 0} chunks for query: "${query}"`);
    const rawChunks = hits.sort((a: any, b: any) => (b.score ?? 1) - (a.score ?? 1)).map((h: any) => h.payload?.text as string);
    return (await this.rerank.rerank(query, rawChunks, 10)).slice(0, 10);
  }

  private parseNumericRange(
    question: string,
    keywords: string[],
  ): { gt?: number; lt?: number; gte?: number; lte?: number } | null {
    const q = question.toLowerCase();
    const kw = keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    const num = '([\\d,]+(?:\\.\\d+)?)\\s*(lakhs?|lacs?|crores?|cr|k|thousand|millions?|mn|billions?|bn)?';
    const re = (ops: string) => new RegExp(`(?:${kw})[^.]*?(?:${ops})\\s*${num}`);
    const val = (m: RegExpMatchArray) => {
      let n = Number(m[1].replace(/,/g, ''));
      const u = m[2];
      if (u) {
        if (/^la(kh|c)s?$/.test(u)) n *= 1e5;
        else if (/^crores?$|^cr$/.test(u)) n *= 1e7;
        else if (/^k$|^thousand$/.test(u)) n *= 1e3;
        else if (/^millions?$|^mn$/.test(u)) n *= 1e6;
        else if (/^billions?$|^bn$/.test(u)) n *= 1e9;
      }
      return n;
    };
    let m: RegExpMatchArray | null;
    if ((m = q.match(re('more than|greater than|over|above|bigger than|>')))) return { gt: val(m) };
    if ((m = q.match(re('at least|minimum|min|>=')))) return { gte: val(m) };
    if ((m = q.match(re('less than|under|below|smaller than|<')))) return { lt: val(m) };
    if ((m = q.match(re('at most|maximum|max|<=')))) return { lte: val(m) };
    if ((m = q.match(re('exactly|equal to|equals|is|are|of|=')))) {
      const n = val(m);
      return { gte: n, lte: n };
    }
    return null;
  }

  private parseCity(question: string): string | null {
    const stop = new Set(['progress', 'process', 'total', 'all', 'the', 'which', 'this', 'that', 'detail', 'details']);
    const m = question.match(
      /\b(?:projects?\s+(?:in|at|from)|located\s+(?:in|at)|based\s+in|city(?:\s+(?:of|is))?)\s+([A-Za-z][A-Za-z .]*?)(?:\s+(?:with|where|which|that|having|and|more|less|greater|area|estimated|value|sqft|sft|having)\b|[?.,]|$)/i,
    );
    if (!m) return null;
    const city = m[1].trim();
    if (city.length < 2 || stop.has(city.toLowerCase())) return null;
    return city;
  }

  private parseType(question: string): 'lead' | 'project' | null {
    if (/\blead(s)?\b/i.test(question)) return 'lead';
    if (/\bproject(s)?\b/i.test(question)) return 'project';
    return null;
  }
}
