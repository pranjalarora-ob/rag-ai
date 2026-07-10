import { Injectable } from '@nestjs/common';
import { Response } from 'express';
import {
  StateGraph,
  Annotation,
  START,
  END,
  MemorySaver,
} from '@langchain/langgraph';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { RerankService } from './rerank.service';
import { GuardrailService } from './guardrail.service';
import { SemanticCacheService } from './semantic-cache.service';
import { PlannerService } from './planner.service';
import { ProjectAgentService } from './project-agent.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';
import { ChatService } from '../chat/chat.service';

/**
 * LangGraph multi-agent orchestrator.
 *
 * This is the "framework" layer requested by stakeholders. It does NOT re-implement
 * any reasoning — it wraps the existing, verified vj-4 engine as graph nodes so the
 * answers stay exactly as correct as the /rag/planner flow:
 *
 *   START
 *    → guardrail    policy check; refuse & END on violation
 *    → cacheCheck   semantic cache (>=0.92); return cached answer & END on hit
 *    → supervisor   LLM router: analytics | lookup | search
 *        ├ analytics → PlannerService      (aggregates / top-N / group-by / charts)
 *        ├ lookup    → ProjectAgentService (exact filtered list/count — deterministic)
 *        └ search    → RAG + rerank + phrase (descriptive lookups)
 *    → cacheSave
 *    → END
 *
 * Generation runs through OpenRouter (the current vj-4 default).
 */
export interface AgentGraphResult {
  answer: string;
  route: string;
  trace: Array<{ tool: string; args: any; result?: any }>;
  cached: boolean;
  sessionId?: string;
}

type Route = 'analytics' | 'lookup' | 'search' | 'schedule';

// Graph state. Kept at module scope so `GraphState` can be referenced in node
// signatures without polymorphic-`this` type gymnastics.
const GraphState = Annotation.Root({
  question: Annotation<string>(),
  customerId: Annotation<string>(),
  history: Annotation<Array<{ role: string; content: string }>>(),
  route: Annotation<Route>(),
  answer: Annotation<string>(),
  trace: Annotation<AgentGraphResult['trace']>(),
  cacheEmbedding: Annotation<number[]>(),
  cached: Annotation<boolean>(),
  blocked: Annotation<boolean>(),
});
type GraphState = typeof GraphState.State;

@Injectable()
export class AgentGraphService {
  private readonly graph: ReturnType<AgentGraphService['buildGraph']>;

  constructor(
    private readonly openai: OpenaiService,
    private readonly qdrant: QdrantService,
    private readonly rerank: RerankService,
    private readonly guardrail: GuardrailService,
    private readonly semanticCache: SemanticCacheService,
    private readonly planner: PlannerService,
    private readonly projectAgent: ProjectAgentService,
    private readonly chatService: ChatService,
  ) {
    this.graph = this.buildGraph();
  }

  // ============ Public API ============

  /** Run the graph to completion and return the final answer + trace. */
  async run(params: {
    question: string;
    customerId: string;
    sessionId?: string;
    history?: Array<{ role: string; content: string }>;
  }): Promise<AgentGraphResult> {
    let activeSessionId = params.sessionId;
    if (!activeSessionId) {
      const session = await this.chatService.createSession(params.customerId, params.question);
      activeSessionId = (session as any)._id.toString();
    }
    // Save user question
    await this.chatService.addMessage(activeSessionId, 'user', params.question);

    const final = await this.graph.invoke(
      {
        question: params.question,
        customerId: params.customerId,
        history: params.history ?? [],
      },
      { configurable: { thread_id: crypto.randomUUID() } },
    );

    const answer = final.answer ?? '';
    // Save assistant answer
    await this.chatService.addMessage(activeSessionId, 'assistant', answer);

    return {
      answer,
      route: final.route ?? 'analytics',
      trace: final.trace ?? [],
      cached: !!final.cached,
      sessionId: activeSessionId,
    };
  }

  /**
   * Streaming variant: runs the same graph, then chunk-streams the final answer to
   * the HTTP response (typing effect), mirroring PlannerService.runStream so the UI
   * behaves identically to /rag/planner/stream.
   */
  async runStream(params: {
    question: string;
    customerId: string;
    sessionId?: string;
    history?: Array<{ role: string; content: string }>;
    res: Response;
    onAnswer?: (answer: string) => void;
  }): Promise<void> {
    const { res, onAnswer } = params;
    const { answer, sessionId } = await this.run(params);
    onAnswer?.(answer);

    if (sessionId) {
      res.setHeader('x-session-id', sessionId);
    }
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.flushHeaders?.();

    const tokens = answer.split(/(\s+)/);
    for (let i = 0; i < tokens.length; i += 3) {
      if (res.writableEnded) break;
      res.write(tokens.slice(i, i + 3).join(''));
      (res as any).flush?.();
      await new Promise((r) => setTimeout(r, 6));
    }
    if (!res.writableEnded) res.end();
  }

  // ============ Nodes ============

  private guardrailNode = async (state: GraphState) => {
    if (this.guardrail.isPolicyViolation(state.question)) {
      console.log('🛡️ Guardrail: policy violation blocked');
      return {
        blocked: true,
        answer: 'This request cannot be processed as it violates company policy.',
      };
    }
    return { blocked: false };
  };

  private cacheCheckNode = async (state: GraphState) => {
    const cache = await this.semanticCache.check(state.question, state.customerId);
    if (cache?.answer) {
      return { cached: true, answer: cache.answer, cacheEmbedding: cache.embedding };
    }
    return { cached: false, cacheEmbedding: cache?.embedding ?? [] };
  };

  private supervisorNode = async (state: GraphState) => {
    const route = await this.classify(state.question);
    console.log(`🧭 Supervisor routed → ${route}`);
    return { route };
  };

  // analytics: aggregates / top-N / group-by / charts — reuse PlannerService (owns projectAnalytics).
  private analyticsAgentNode = async (state: GraphState) => {
    const r = await this.planner.run({
      question: state.question,
      customerId: state.customerId,
      history: state.history,
    });
    return { answer: r.answer, trace: r.trace };
  };

  // lookup: exact filtered list/count by field — reuse ProjectAgentService (deterministic queryProjects).
  private lookupAgentNode = async (state: GraphState) => {
    const r = await this.projectAgent.run({
      question: state.question,
      customerId: state.customerId,
    });

    // For an exact single-project lookup, ProjectAgentService returns the FULL record
    // text. If the user asked a focused question ("what is the owner?"), extract just the
    // relevant part from that exact record — grounded, so nothing is invented. Tables /
    // counts (contain "|") stay raw and deterministic; we never re-phrase those.
    const singleRecord =
      !!r.answer &&
      !r.answer.includes('|') &&
      r.answer !== 'No projects match that filter.' &&
      r.trace?.some((t) => t.tool === 'queryProjects' && t.args?.projectCode);

    // Broad "tell me everything" questions about one project → return a structured
    // card block the UI renders as a ProjectCard (deterministic, built from the DB
    // payload — not LLM prose). Narrow questions ("who is the owner?") skip the card
    // and get a focused text answer instead.
    const broadIntent =
      /\b(info|information|details?|tell me about|about|summary|overview|everything|show|profile)\b/i.test(
        state.question,
      );
    const record = r.rows?.[0];
    if (singleRecord && broadIntent && record) {
      const block = this.buildProjectCardBlock(record);
      if (block) return { answer: block, trace: r.trace };
    }

    if (singleRecord) {
      const focused = await this.openai.openRouterGenerate(
        `${SYSTEM_PROMPT}

Answer the user's question CONCISELY using ONLY the exact project record below.
Give just the detail asked for — do not list unrelated fields. If the detail is missing, say so.

RECORD:
${r.answer}

QUESTION: ${state.question}`,
        0,
      );
      if (focused?.trim()) return { answer: focused.trim(), trace: r.trace };
    }

    return { answer: r.answer, trace: r.trace };
  };

  // search: descriptive/semantic lookup — RAG + rerank + phrase.
  private searchAgentNode = async (state: GraphState) => {
    const embedding = await this.openai.generateEmbedding(state.question);
    const hits = await this.qdrant.search(COLLECTION, {
      vector: embedding,
      limit: 50,
      filter: { must: [{ key: 'customerId', match: { value: state.customerId } }] },
    });
    const rawChunks = hits
      .sort((a: any, b: any) => b.score - a.score)
      .map((h: any) => h.payload?.text as string)
      .filter(Boolean);
    const reranked = (await this.rerank.rerank(state.question, rawChunks, 10)).slice(0, 10);

    if (!reranked.length) {
      return {
        answer: "I don't have information about that in the project records.",
        trace: [{ tool: 'searchProjects', args: { query: state.question }, result: [] }],
      };
    }

    const context = reranked.join('\n\n---\n\n');
    const answer = await this.openai.openRouterGenerate(
      `${SYSTEM_PROMPT}

Answer the user's question using ONLY the project context below. Be concise.

CONTEXT:
${context}

QUESTION: ${state.question}`,
      0.2,
    );

    return {
      answer: answer || 'I could not find a relevant answer in the project records.',
      trace: [{ tool: 'searchProjects', args: { query: state.question }, result: reranked }],
    };
  };

  // schedule: project workflow / timeline. Deterministically pulls the project's
  // flow phases (docType 'project-flow-phase') and returns a structured block the
  // UI renders as a stepper. Not LLM-phrased — dates/statuses come straight from data.
  private scheduleAgentNode = async (state: GraphState) => {
    const code = this.extractProjectCode(state.question, state.history);
    if (!code) {
      return { answer: 'Which project would you like the schedule for? Please include its project code.' };
    }

    const must: any[] = [
      { key: 'customerId', match: { value: state.customerId } },
      { key: 'docType', match: { value: 'project-flow-phase' } },
    ];
    if (code) {
      must.push({ key: 'projectCode', match: { value: String(code) } });
    }
    const points = await this.qdrant.scrollAll(COLLECTION, { must });

    const seen = new Set<string>();
    const phases: any[] = [];
    for (const p of points) {
      const pl: any = p.payload || {};
      const id = pl.phaseId || pl.original_id;
      if (id && seen.has(id)) continue;
      if (id) seen.add(id);
      phases.push(pl);
    }
    if (!phases.length) {
      return {
        answer: `No project schedule found for project ${code}.`,
        trace: [{ tool: 'projectSchedule', args: { projectCode: code }, result: [] }],
      };
    }
    phases.sort((a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0));

    const name = phases[0].projectName || `Project ${code}`;
    return {
      answer: this.buildScheduleBlock(name, code, phases),
      trace: [{ tool: 'projectSchedule', args: { projectCode: code }, result: phases.length }],
    };
  };

  private cacheSaveNode = async (state: GraphState) => {
    if (!state.cached && !state.blocked && state.answer && state.cacheEmbedding?.length) {
      this.semanticCache
        .save(state.question, state.cacheEmbedding, state.answer, state.customerId)
        .catch(() => { });
    }
    return {};
  };

  // Build a fenced ```project-card block from a project payload. The UI parses this
  // JSON and renders a rich ProjectCard; other clients just see a JSON code block.
  private buildProjectCardBlock(pl: any): string | null {
    if (!pl) return null;
    const name =
      pl.projectName || (pl.companyName || '').trim() || String(pl.projectCode || 'Project');
    const num = (x: any) => {
      const n = Number(x);
      return Number.isFinite(n) && n > 0 ? n : undefined;
    };
    const card = {
      code: pl.projectCode != null ? String(pl.projectCode) : undefined,
      name,
      type: pl.type || undefined,
      status: pl.projectStatus || pl.leadStatus || undefined,
      leadStatus: pl.leadStatus || undefined,
      stage: [pl.stage, pl.subStage].filter(Boolean).join(' / ') || undefined,
      owner: pl.owner || undefined,
      city: [pl.city, pl.state].filter(Boolean).join(', ') || undefined,
      zone: pl.zone || undefined,
      area: num(pl.areaSft),
      scope: pl.scope || undefined,
      channel: pl.channel || undefined,
      priority: pl.priority != null ? String(pl.priority) : undefined,
      estimatedValue: num(pl.estimatedValue),
      currentProjectValue: num(pl.currentProjectValue),
      closureValue: num(pl.closureValue),
      customer: pl.customerInfo?.name
        ? {
          name: pl.customerInfo.name,
          email: pl.customerInfo.email || undefined,
          mobile: pl.customerInfo.mobile || undefined,
        }
        : undefined,
      team: Array.isArray(pl.team)
        ? pl.team
          .map((m: any) => {
            const name = m.name || '';
            const role = m.role || m.pocRole || '';
            return name && role ? `${name} (${role})` : (name || role);
          })
          .filter(Boolean)
          .slice(0, 6)
        : undefined,
      projectId: pl.projectId || undefined,
    };
    // Drop undefined keys so the payload stays compact.
    const clean = JSON.parse(JSON.stringify(card));
    return '```project-card\n' + JSON.stringify(clean) + '\n```';
  }

  // Pull a project code (6+ digit number) from the question, or fall back to the
  // most recent code mentioned earlier in the conversation — so a bare follow-up
  // like "Project schedule" resolves against the project just discussed.
  private extractProjectCode(question: string, history?: Array<{ role: string; content: string }>): string | null {
    const m = question.match(/\b(\d{6,})\b/);
    if (m) return m[1];
    for (const h of [...(history || [])].reverse()) {
      const hm = String(h?.content || '').match(/\b(\d{6,})\b/);
      if (hm) return hm[1];
    }
    return null;
  }

  // Build a fenced ```project-schedule block from the project's flow phases. The UI
  // parses this JSON and renders a stepper (phases + nested milestones with status
  // and dates); other clients just see a JSON code block.
  private buildScheduleBlock(name: string, code: string, phases: any[]): string {
    const clean = {
      code,
      name,
      phases: phases.map((p) => ({
        name: p.name,
        code: p.code || undefined,
        status: p.status || null,
        sequence: p.sequence,
        startDate: p.startDate || null,
        endDate: p.endDate || null,
        milestones: (Array.isArray(p.milestones) ? p.milestones : [])
          .slice()
          .sort((a: any, b: any) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0))
          .map((m: any) => ({
            name: m.name,
            status: m.status || null,
            sequence: m.sequence,
            startDate: m.startDate || null,
            endDate: m.endDate || null,
          })),
      })),
    };
    return '```project-schedule\n' + JSON.stringify(clean) + '\n```';
  }

  // ============ Routing helpers ============

  private async classify(question: string): Promise<Route> {
    // Deterministic disambiguation first — a specific field filter can't be an aggregate.
    const q = question.toLowerCase();
    // Schedule/workflow intent → the project-flow stepper. Wins over other routes.
    const scheduleIntent = /\b(schedule|timeline|road ?map|workflow|project flow|phases?|milestones?)\b/.test(q);
    if (scheduleIntent) return 'schedule';
    // If the question contains a 6+ digit project code, bypass the LLM and route directly to lookup!
    if (/\b(\d{6,})\b/.test(q)) return 'lookup';
    const descriptive = /\b(tell me about|who is|who are|design manager|notes|describe|context|details about)\b/.test(q);
    const fieldFilter = /\b(in|from|at)\s+[a-z]|area|sqft|sq ft|estimated value|owner|zone|project code|leads?\b|projects?\b/.test(q);
    const aggregate = /\b(total|sum|average|avg|count|how many|highest|lowest|top\s*\d+|rank|most|least|combined|per\s+\w+|group)\b/.test(q);

    const raw = (
      await this.openai.openRouterGenerate(
        `Classify the question into ONE route word: analytics, lookup, or search.
- analytics: totals, sums, averages, counts, "how many", top-N rankings, group-by, charts/comparisons across many projects.
- lookup: exact filtered list or count by a specific field (city, owner, zone, area, estimated value, project code, lead vs project).
- search: descriptive/semantic question about a specific project (notes, who is on the team, "tell me about", context).
Return ONLY the single route word, nothing else.

Question: "${question}"`,
        0,
      )
    )
      .trim()
      .toLowerCase();

    const llmRoute: Route | null =
      raw.includes('analytics') ? 'analytics' :
        raw.includes('lookup') ? 'lookup' :
          raw.includes('search') ? 'search' : null;

    // Trust the LLM, but let strong deterministic signals override obvious misroutes.
    if (descriptive && !aggregate) return 'search';
    if (aggregate) return 'analytics';
    if (llmRoute) return llmRoute;
    if (fieldFilter) return 'lookup';
    return 'analytics';
  }

  private routeDecision = (state: GraphState): Route => state.route ?? 'analytics';

  // ============ Graph assembly ============

  private buildGraph() {
    const workflow = new StateGraph(GraphState)
      .addNode('guardrail', this.guardrailNode)
      .addNode('cacheCheck', this.cacheCheckNode)
      .addNode('supervisor', this.supervisorNode)
      .addNode('analytics', this.analyticsAgentNode)
      .addNode('lookup', this.lookupAgentNode)
      .addNode('search', this.searchAgentNode)
      .addNode('schedule', this.scheduleAgentNode)
      .addNode('cacheSave', this.cacheSaveNode)
      .addEdge(START, 'guardrail')
      .addConditionalEdges('guardrail', (s) => (s.blocked ? END : 'cacheCheck'), {
        cacheCheck: 'cacheCheck',
        [END]: END,
      })
      .addConditionalEdges('cacheCheck', (s) => (s.cached ? END : 'supervisor'), {
        supervisor: 'supervisor',
        [END]: END,
      })
      .addConditionalEdges('supervisor', this.routeDecision, {
        analytics: 'analytics',
        lookup: 'lookup',
        search: 'search',
        schedule: 'schedule',
      })
      .addEdge('analytics', 'cacheSave')
      .addEdge('lookup', 'cacheSave')
      .addEdge('search', 'cacheSave')
      .addEdge('schedule', 'cacheSave')
      .addEdge('cacheSave', END);

    return workflow.compile({ checkpointer: new MemorySaver() });
  }
}
