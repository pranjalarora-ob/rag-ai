import { Injectable } from '@nestjs/common';
import { Response } from 'express';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { RerankService } from './rerank.service';
import { ClaudeService } from './claude.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

const MAX_STEPS = 5; // safety cap on the tool→evaluate→tool loop

// Appended to the planner's system prompt so list-style answers render as tables
// the UI can turn into charts. The chart parser needs a "Name" column plus a
// "Value"/"Area" column, so the columns below must stay in sync with it.
const TABLE_FORMAT_INSTRUCTION = `
=== CRITICAL OUTPUT RULES — MUST FOLLOW EXACTLY ===

RULE 1 — NEVER say you cannot draw a chart. The UI draws charts FOR you from tables.
RULE 2 — NEVER use bullet points or numbered lists when showing multiple projects.
RULE 3 — ALWAYS use a markdown table when showing 2+ projects. Exact columns:
  | # | Code | Name | City | Area (sqft) | Estimated Value |
  Use RAW integers in Area and Estimated Value (NO commas, NO "Cr"/"L" suffix, NO units).

RULE 4 — When the user asks for a "chart", "graph", "pie", "bar", "comparison", or "visualize":
  → Call projectAnalytics with metric="area" (or relevant metric) and topN=50
  → Return the results as the markdown table above
  → Do NOT say anything like "I am unable to provide a chart"

Example of CORRECT response to "show top 5 projects by area":
| # | Code | Name | City | Area (sqft) | Estimated Value |
|---|------|------|------|-------------|-----------------|
| 1 | P001 | Alpha Tower | Delhi | 85000 | 50000000 |
| 2 | P002 | Beta Mall | Mumbai | 72000 | 43000000 |

RULE 5 — Only use plain prose (no table) for single facts, yes/no, or counts with no row data.
=== END CRITICAL RULES ===`;

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
    private readonly claudeService: ClaudeService,
  ) { }

  private readonly tools = [
    {
      type: 'function',
      function: {
        name: 'searchProjects',
        description:
          'Search project records semantically by meaning. You can also apply exact filters if the user question specifies them. Let the LLM decide which filters should apply.',
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
    {
      type: 'function',
      function: {
        name: 'projectAnalytics',
        description:
          'Compute EXACT aggregates over ALL of the customer\'s projects: totals, averages, counts, top-N rankings, filtered sums/lists. Use for any "total", "how many", "top N", "highest/lowest", "average", or "where <person> is <role>" question. Never compute numbers yourself — call this.',
        parameters: {
          type: 'object',
          properties: {
            metric: { type: 'string', enum: ['estimatedValue', 'boqValue', 'area'], description: 'What to rank/total/chart by. Use "area" for area-based questions ("compare their area", "largest by area").' },
            topN: { type: 'number', description: 'Return only the top N by value/area. Set to 50 when asked to chart/compare all results or when a previous list was longer than 5.' },
            teamMember: { type: 'string', description: 'Filter to projects where this person is on the team.' },
            role: { type: 'string', description: 'Role to match with teamMember, e.g. "Design Manager".' },
            groupBy: {
              type: 'string',
              enum: ['stage', 'subStage', 'city', 'state', 'zone', 'owner', 'channel', 'projectStatus'],
              description: 'Group counts/totals by this field, e.g. "how many projects per stage".',
            },
            city: { type: 'string', description: 'Filter by city, e.g. "Gurugram".' },
            state: { type: 'string', description: 'Filter by state, e.g. "Haryana".' },
            zone: { type: 'string', description: 'Filter by zone, e.g. "Gurgaon".' },
            stage: { type: 'string', description: 'Filter by stage, e.g. "Execution", "Design-Sales".' },
            subStage: { type: 'string', description: 'Filter by sub-stage, e.g. "Handover".' },
            owner: { type: 'string', description: 'Filter by owner/team, e.g. "SMB Team".' },
            channel: { type: 'string', description: 'Filter by channel, e.g. "Digital".' },
            projectStatus: { type: 'string', description: 'Filter by project status, e.g. "Cancelled", "InProgress".' },
            companyName: { type: 'string', description: 'Filter by company/project company name, e.g. "Officebanao".' },
            customerName: { type: 'string', description: 'Filter by the customer contact name.' },
            minValue: { type: 'number', description: 'estimatedValue >= this (in rupees). Convert "2 cr" -> 20000000, "40 lakh" -> 4000000.' },
            maxValue: { type: 'number', description: 'estimatedValue <= this (in rupees).' },
            minArea: { type: 'number', description: 'areaSft >= this (in sqft).' },
            maxArea: { type: 'number', description: 'areaSft <= this (in sqft).' },
          },
        },
      },
    },
  ];

  private buildSystemInstruction(customerId: string): string {
    return `${SYSTEM_PROMPT}

${TABLE_FORMAT_INSTRUCTION}

You are the PLANNER. Decide which tool(s) to call to answer the user, call them, read the results, then write the final answer.
Rules:
- The customerId is "${customerId}" and is already known — never ask the user for it.
- For totals, counts, averages, rankings, filtered sums/lists, or ANY chart/comparison request, you MUST call projectAnalytics. Do not calculate numbers yourself.
- For descriptive questions about a specific project, call searchProjects.
- Base every fact and number ONLY on tool results. If the tools return nothing relevant, say you don't have that information.
- When you have enough information, reply with the final answer as a markdown table (if multiple records) or plain text (if single fact).`;
  }

  // Run the tool-calling loop until the model has everything it needs to answer.
  // Returns { messages, trace, done }: `done` is the final answer text if the model
  // produced one without needing a final streamed turn, else null. `messages` holds
  // the full conversation (incl. tool results) ready for a final answer turn.
  private async resolveTools(
    messages: any[],
    customerId: string,
    originalQuestion: string,
  ): Promise<{ trace: PlannerResult['trace']; finalText: string | null; steps: number }> {
    const trace: PlannerResult['trace'] = [];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const assistant = this.claudeService.isConfigured
        ? await this.claudeService.claudeToolTurn(messages, this.tools)
        : await this.openaiService.openRouterToolTurn(messages, this.tools);
      const toolCalls = assistant?.tool_calls;
      console.log("toolCalls", toolCalls);

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
        const result = await this.executeTool(call.function?.name, args, customerId, originalQuestion);
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

    const { trace, finalText, steps } = await this.resolveTools(messages, customerId, question);
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
    onAnswer?: (answer: string) => void;
  }): Promise<void> {
    const { question, customerId, history = [], res, onAnswer } = params;
    console.log("runStream", question);
    const messages: any[] = [
      { role: 'system', content: this.buildSystemInstruction(customerId) },
      ...history,
      { role: 'user', content: question },
    ];

    // Drive the tool loop. Its final turn already produces the answer text, so we
    // reuse that instead of re-generating it with a second streaming LLM call —
    // saving one full generation round-trip per request.
    const { finalText } = await this.resolveTools(messages, customerId, question);
    const answer = finalText || 'I could not complete the request within the allowed number of steps.';
    console.log("finalText", finalText)
    onAnswer?.(answer);

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

  private async executeTool(name: string, args: any, customerId: string, originalQuestion?: string) {
    try {
      if (name === 'searchProjects') {
        const queryStr = args.query || '';
        const embedding = await this.openaiService.generateEmbedding(queryStr);

        const filterMust: any[] = [{ key: 'customerId', match: { value: customerId } }];

        // 1. projectCode
        const projectCode = args.projectCode || queryStr.match(/\b(\d{6,})\b/)?.[1] || (originalQuestion || '').match(/\b(\d{6,})\b/)?.[1];
        if (projectCode) {
          filterMust.push({
            key: 'projectCode',
            match: { value: String(projectCode) }
          });
        }

        // 2. city
        let city = args.city;
        if (!city && !projectCode) {
          city = this.parseCity(queryStr);
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
          typeFilter = this.parseType(queryStr);
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
          const isFlowPhaseQuery = /phase|milestone|stage|workflow/i.test(queryStr) ||
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
          let areaRange = this.parseNumericRange(queryStr, ['area', 'sqft', 'sft', 'square feet', 'square foot']);
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
          let valueRange = this.parseNumericRange(queryStr, ['estimated value', 'estimatedvalue', 'value', 'budget', 'worth']);
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
          hits = await this.qdrantService.scrollAll(COLLECTION, { must: filterMust });
        } else {
          hits = await this.qdrantService.search(COLLECTION, {
            vector: embedding,
            limit: 10,
            filter: { must: filterMust },
          });
        }
        console.log(`[Qdrant Search] Received ${hits?.length || 0} chunks for query: "${queryStr}"`);
        console.log("hitshits", hits);
        const rawChunks = hits
          .sort((a: any, b: any) => (b.score ?? 1) - (a.score ?? 1))
          .map((h: any) => h.payload?.text as string);
        const reranked = await this.rerankService.rerank(queryStr, rawChunks, 10);
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
          state: args.state,
          zone: args.zone,
          stage: args.stage,
          subStage: args.subStage,
          owner: args.owner,
          channel: args.channel,
          projectStatus: args.projectStatus,
          companyName: args.companyName,
          customerName: args.customerName,
          minValue: args.minValue,
          maxValue: args.maxValue,
          minArea: args.minArea,
          maxArea: args.maxArea,
        });
      }

      return { error: `Unknown tool: ${name}` };
    } catch (err: any) {
      console.error('executeTool error:', err);
      return { error: err?.message || 'tool execution failed' };
    }
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
