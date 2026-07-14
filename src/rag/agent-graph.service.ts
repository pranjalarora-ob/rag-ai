import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { Response } from 'express';
import {
  StateGraph,
  Annotation,
  START,
  END,
  MemorySaver,
  messagesStateReducer,
} from '@langchain/langgraph';
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage
} from '@langchain/core/messages';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { RerankService } from './rerank.service';
import { GuardrailService } from './guardrail.service';
import { SemanticCacheService } from './semantic-cache.service';
import { PlannerService } from './planner.service';
import { ProjectAgentService } from './project-agent.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';
import { ChatService } from '../chat/chat.service';
import axios from 'axios';
import { ConfigService } from '@nestjs/config';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

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
  userName: Annotation<string>(),
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
  // Optional: only built when GEMINI_API_KEY is present, so a missing key never
  // crashes the live app on boot.
  private readonly graphV10?: ReturnType<AgentGraphService['agentGraphV10']>;
  private readonly OPEN_ROUTER_API_KEY: string;
  private readonly model!: ChatGoogleGenerativeAI;
  private readonly GEMINI_API_KEY: string;

  constructor(
    private readonly openai: OpenaiService,
    private readonly qdrant: QdrantService,
    private readonly rerank: RerankService,
    private readonly guardrail: GuardrailService,
    private readonly semanticCache: SemanticCacheService,
    private readonly planner: PlannerService,
    private readonly projectAgent: ProjectAgentService,
    private readonly chatService: ChatService,
    private readonly configService: ConfigService,
    
  ) {
    this.graph = this.buildGraph();
    this.OPEN_ROUTER_API_KEY = this.configService.get('OPEN_ROUTER_KEY') || '';
    this.GEMINI_API_KEY = this.configService.get('GEMINI_API_KEY') || '';
    // V10 runs on Gemini. Only build the model + V10 graph when a key is configured —
    // ChatGoogleGenerativeAI throws at construction without one, which would otherwise
    // crash the whole service (and take down the live /agent-graph routes too).
    if (this.GEMINI_API_KEY) {
      this.model = new ChatGoogleGenerativeAI({
        model: 'gemini-2.5-flash',
        apiKey: this.GEMINI_API_KEY,
        temperature: 0.2,
        maxOutputTokens: 2048,
      });
      this.graphV10 = this.agentGraphV10();
    } else {
      console.warn('[AgentGraph] GEMINI_API_KEY not set — V10 endpoints are disabled (live graph unaffected).');
    }
  }

  // ============ Public API ============

  /** Run the graph to completion and return the final answer + trace. */
  async run(params: {
    question: string;
    customerId: string;
    sessionId?: string;
    history?: Array<{ role: string; content: string }>;
    userName?: string;
  }): Promise<AgentGraphResult> {
    let activeSessionId = params.sessionId;
    if (!activeSessionId) {
      const session = await this.chatService.createSession(params.customerId, params.question);
      activeSessionId = (session as any)._id.toString();
    }
    // Persist the user message but DON'T block the response on the DB write.
    this.chatService.addMessage(activeSessionId, 'user', params.question).catch(() => {});

    // Fast-paths that don't need the graph: identity ("who am I") and small-talk.
    const canned = this.identityAnswer(params.question, params.userName) ?? this.smallTalkAnswer(params.question);
    let answer: string;
    let route = 'smalltalk';
    let trace: AgentGraphResult['trace'] = [];
    let cached = false;

    if (canned) {
      answer = canned;
    } else {
      const final = await this.graph.invoke(
        {
          question: params.question,
          customerId: params.customerId,
          history: params.history ?? [],
          userName: params.userName ?? '',
        },
        { configurable: { thread_id: crypto.randomUUID() } },
      );
      answer = final.answer ?? '';
      route = final.route ?? 'analytics';
      trace = final.trace ?? [];
      cached = !!final.cached;
    }

    // Persist the assistant answer without blocking either.
    this.chatService.addMessage(activeSessionId, 'assistant', answer).catch(() => {});

    return { answer, route, trace, cached, sessionId: activeSessionId };
  }

  // Identity fast-path: "who am I" / "what's my name" → answer from the passed-in
  // user name, no retrieval. Returns null if there's no name or it isn't an identity Q.
  private identityAnswer(question: string, userName?: string): string | null {
    if (!userName) return null;
    const q = (question || '').trim().toLowerCase().replace(/[!.?,]+$/g, '').trim();
    if (/^(who am i|what('?s| is) my name|do you know (who i am|my name)|my name)$/.test(q)) {
      return `You're ${userName}. I can show your projects, leads, or schedules — for example ask "my projects".`;
    }
    return null;
  }

  // Whole-message small talk → a canned reply, so trivial inputs skip the pipeline.
  // Matches the ENTIRE message (not a substring), so "hi, cost of project X" still
  // goes through the graph — only a bare "hi" / "thanks" / "ok" short-circuits.
  private smallTalkAnswer(question: string): string | null {
    const q = (question || '').trim().toLowerCase().replace(/[!.?,]+$/g, '').trim();
    if (!q || q.length > 24) return null;
    const greet = ['hi', 'hii', 'hello', 'helo', 'hey', 'heya', 'yo', 'hola', 'namaste', 'good morning', 'good afternoon', 'good evening'];
    const thank = ['thanks', 'thank you', 'thankyou', 'thank u', 'thx', 'ty'];
    const bye = ['bye', 'goodbye', 'see you', 'see ya', 'cya'];
    const ack = ['ok', 'okay', 'okk', 'k', 'cool', 'great', 'nice', 'got it', 'sounds good', 'perfect', 'awesome'];
    if (greet.includes(q)) return 'Hi! I can help with project info — ask me about a project, its cost, area, stage, zone, schedule, or leads.';
    if (thank.includes(q)) return "You're welcome! Anything else you'd like to know about your projects?";
    if (bye.includes(q)) return 'Goodbye! Come back anytime you need project information.';
    if (ack.includes(q)) return 'Got it. What would you like to know about your projects?';
    return null;
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
    userName?: string;
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

  // ============ V10 (experimental) — separate from the live graph ============

  /**
   * Run the experimental V10 graph (Gemini generation + full-dataset aggregation).
   * Kept fully separate from run()/runStream() so the live version is never affected.
   * Persists to chat history the same way; returns the final answer text.
   */
  async runV10(params: {
    question: string;
    customerId: string;
    sessionId?: string;
    userName?: string;
  }): Promise<AgentGraphResult> {
    if (!this.graphV10) {
      return {
        answer: 'V10 is not configured on this environment (missing GEMINI_API_KEY). The standard assistant is available at /rag/agent-graph.',
        route: 'v10-disabled',
        trace: [],
        cached: false,
      };
    }
    let activeSessionId = params.sessionId;
    if (!activeSessionId) {
      const session = await this.chatService.createSession(params.customerId, params.question);
      activeSessionId = (session as any)._id.toString();
    }
    this.chatService.addMessage(activeSessionId, 'user', params.question).catch(() => {});

    let answer: string;
    const canned = this.identityAnswer(params.question, params.userName) ?? this.smallTalkAnswer(params.question);
    if (canned) {
      answer = canned;
    } else {
      const final = await this.graphV10.invoke(
        { question: params.question, customerId: params.customerId },
        { configurable: { thread_id: crypto.randomUUID() } },
      );
      answer = (final as any).result ?? (final as any).answer ?? 'I could not complete the request.';
    }

    this.chatService.addMessage(activeSessionId, 'assistant', answer).catch(() => {});
    return { answer, route: 'v10', trace: [], cached: false, sessionId: activeSessionId };
  }

  /** Streaming variant of the V10 graph — mirrors runStream()'s chunked output. */
  async runV10Stream(params: {
    question: string;
    customerId: string;
    sessionId?: string;
    userName?: string;
    res: Response;
    onAnswer?: (answer: string) => void;
  }): Promise<void> {
    const { res, onAnswer } = params;
    const { answer, sessionId } = await this.runV10(params);
    onAnswer?.(answer);

    if (sessionId) res.setHeader('x-session-id', sessionId);
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
    if (this.guardrail.isAbusive(state.question)) {
      console.log('🛡️ Guardrail: abusive language blocked');
      return {
        blocked: true,
        answer:
          "I'd like to keep our conversation respectful. I'm happy to help with any questions about your projects — their details, stages, schedules, or numbers.",
      };
    }
    if (this.guardrail.isOutOfScopeAction(state.question)) {
      console.log('🛡️ Guardrail: out-of-scope action blocked');
      return {
        blocked: true,
        answer:
          "That's outside what I can do — I'm a read-only assistant for looking up project information (details, counts, stages, schedules, leads). I can't create, edit, or delete records. Please use the Workbench for that.",
      };
    }
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
      // For "above/these/those" follow-ups, hand the planner the exact codes from the
      // previous list so the (weak) LLM doesn't have to scrape them out of the table.
      referencedCodes: this.extractRecentCodes(state.question, state.history),
      userName: state.userName,
    });
    return { answer: r.answer, trace: r.trace };
  };

  // If the question is referential ("above/these/those projects"), pull the project
  // codes from the most recent assistant list in the conversation. Returns [] otherwise.
  private extractRecentCodes(question: string, history?: Array<{ role: string; content: string }>): string[] {
    const referential = /\b(above|these|those|them|aforementioned|the (above|previous|listed)|that list|same (projects?|ones))\b/i.test(question);
    if (!referential || !history?.length) return [];
    for (const h of [...history].reverse()) {
      const codes = String(h?.content || '').match(/\b\d{6,}\b/g);
      if (codes && codes.length) return [...new Set(codes)].slice(0, 50);
    }
    return [];
  }

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
    const hasCode = /\b(\d{6,})\b/.test(q);
    // Referential follow-up ("owner of the above projects", "details of these",
    // "who owns them") — must resolve against the previous list in the conversation.
    // Only the planner receives history, so route these there (lookup ignores history).
    const referential =
      /\b(above|these|those|them|aforementioned|the (above|previous|listed)|that list|same (projects?|ones))\b/.test(q);
    if (referential) return 'analytics';
    // Schedule/workflow intent → the single-project flow stepper. Only when a specific
    // project code is present: "schedule/timeline/milestones for project 2022072258".
    const scheduleIntent = /\b(schedule|timeline|road ?map|workflow|project flow|phases?|milestones?)\b/.test(q);
    if (scheduleIntent && hasCode) return 'schedule';
    // Aggregate workflow/stage questions across projects (no single code) — pending
    // payments, stuck-at-stage, handover milestones, mobilization, funnel — go to the
    // planner (analytics), which owns the queryProjectFlow tool.
    const flowIntent = /\b(milestones?|payment (due|pending)|pending payment|overdue|due (in|within|this|next)|handover|handed over|mobili[sz]ation|behind schedule|stuck (at|in)|pending (approval|action)|awaiting (client )?approval|site not started|funnel|stage)\b/.test(q);
    if (flowIntent && !hasCode) return 'analytics';
    // Temporal / trend / lead / group-by intent → the planner owns the date + lead +
    // groupBy tools, so route these there rather than to lookup (which has none).
    // e.g. "new leads last 3 months", "zone-wise ...", "trend over 6 months".
    const temporalTrendIntent =
      /\b(last|past)\s+\d+\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)\b|\b(this|previous|current|next)\s+(week|month|quarter|year)\b|\btrend\b|\bnew leads?\b|\bunassigned\b|\bactive\b|[a-z]+-wise\b|\bwise\b/.test(q);
    if (temporalTrendIntent && !hasCode) return 'analytics';
    // Person / role queries — "projects of <name>", "where X is a team member/customer/GM",
    // per-GM workload. The planner owns teamMember/customerName/groupByRole.
    const personIntent =
      /\b(team member|is (the|a) (customer|gm|owner|team member|general manager)|as (a )?(team member|customer|gm|owner)|owned by|handled by|managed by|per gm|gm workload|which gm)\b/i.test(q) ||
      /\bprojects?\s+(of|for|by|under)\s+[a-z]/i.test(q); // "projects of Nitish"
    if (personIntent && !hasCode) return 'analytics';
    // First-person ("my projects", "leads assigned to me") → planner resolves "me" to
    // the logged-in user name.
    const firstPersonIntent =
      /\bmy\s+(projects?|leads?)\b|\b(projects?|leads?)\b[^.]*\b(assigned to me|of mine|i'?m on)\b|\bassigned to me\b/i.test(q);
    if (firstPersonIntent && !hasCode) return 'analytics';
    // If the question contains a 6+ digit project code, bypass the LLM and route directly to lookup!
    if (hasCode) return 'lookup';
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

  async normalizeSemanticQuestion(
    text: string,
  ): Promise<string> {
    const stopWords = new Set([
      'what',
      'which',
      'does',
      'do',
      'is',
      'are',
      'the',
      'a',
      'an',
      'how',
      'why',
      'when',
      'where',
      'who',
      'kind',
      'type',
      'of',
      'to',
      'and',
      'or',
      'in',
      'on',
      'for',
      'this',
      'that',
      'these',
      'those',
    ]);

    const synonymMap: Record<
      string,
      string
    > = {
      meeting: 'interview',
      discussion: 'interview',

      takeaways: 'points',
      takeaway: 'points',

      summary: 'overview',
    };

    const words =
      text
        .toLowerCase()
        .match(/\b[a-z0-9-]+\b/g) || [];

    const normalizedWords =
      words
        .filter(
          word => !stopWords.has(word),
        )
        .map(
          word =>
            synonymMap[word] || word,
        );

    return normalizedWords.join(' ');
  }

  async generateGeminiEmbedding(input: string): Promise<number[]> {
    try {

      // const response = await axios.post(`${this.GEMINI_EMBEDDING_URL}?key=${this.GEMINI_API_KEY}`, {
      //   // model: 'models/text-embedding-004',
      //   content: { parts: [{ text: input }] },
      //   taskType: 'RETRIEVAL_DOCUMENT',
      //   // outputDimensionality: 1536,
      //   outputDimensionality: 3072,
      // });

      const response = await axios.post(
        'https://openrouter.ai/api/v1/embeddings',
        {
          model: 'google/gemini-embedding-001',
          input,
          dimensions: 1536,
          encoding_format: 'float',
        },
        {
          headers: {
            Authorization: `Bearer ${this.OPEN_ROUTER_API_KEY}`,
            'Content-Type': 'application/json',
          },
        },
      );
      // console.log('embeding output', response)
      const embedding = response.data.data?.[0]?.embedding;


      // const embedding = response.data.embedding?.values;

      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('No embedding values returned from Gemini');
      }

      return embedding;
    } catch (error) {
      console.error('Gemini Embedding Error:', error);
      throw new InternalServerErrorException(`Failed to generate Gemini embedding: ${error.message}`);
    }
  }

  private agentGraphV10() {
    // agentGraphV10 is intentionally self-contained: all workflow helpers remain local to this method.
    // NEW IN V10: "aggregation" route — for count/breakdown/compare/trend questions that need
    // the FULL matching dataset (via Qdrant scroll), not a top-k similarity search.
    // ============ 1. GRAPH ANNOTATION ============

    const GraphAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
      }),
      question: Annotation<string>(),
      normalizedQuestion: Annotation<string>(),
      embedding: Annotation<number[]>(),
      route: Annotation<string>(),
      customerId: Annotation<string>(),
      context: Annotation<string>(),
      summary: Annotation<string>(),
      costAnalysis: Annotation<string>(),
      zoneAnalysis: Annotation<string>(),
      flowAnalysis: Annotation<string>(),
      aggregationResult: Annotation<string>(),
      result: Annotation<string>(),
      metadata: Annotation<any>(),
      subQuestions: Annotation<string[]>(),
    });

    type GraphState = typeof GraphAnnotation.State;

    const normalizeQuestionText = (question: string): string =>
      (question || "").trim().replace(/\s+/g, " ");

    const splitCompoundQuestions = (question: string): string[] => {
      const normalizedQuestion = normalizeQuestionText(question);
      if (!normalizedQuestion) {
        return [];
      }

      const sentenceParts = normalizedQuestion
        .split(/(?<=[?.!])\s+/)
        .map((part) => part.trim())
        .filter(Boolean);

      if (sentenceParts.length > 1) {
        return sentenceParts
          .map((part) => part.replace(/[?!.]+$/g, "").trim())
          .filter(Boolean);
      }

      const connectorPattern = /(?:\s+(?:and|also|plus|along with|together with|as well as|then|next)\s+)/i;
      const splitParts = normalizedQuestion
        .split(connectorPattern)
        .map((part) => part.trim())
        .filter(Boolean);

      if (splitParts.length > 1) {
        const hasFollowUpIntent = splitParts.some((part) =>
          /\b(what|which|when|where|who|how|show|compare|analyze|give|tell|list|explain|status|cost|phase|stage|substage|milestone|timeline|progress|budget|flow|count|many)\b/i.test(part),
        );

        if (hasFollowUpIntent) {
          return splitParts
            .map((part) => part.replace(/[?!.]+$/g, "").trim())
            .filter(Boolean);
        }
      }

      return [normalizedQuestion];
    };

    type RouteType =
      | "rag"
      | "summary"
      | "cost"
      | "zone_analysis"
      | "flow_analysis"
      | "aggregation";

    // ============ NEW: AGGREGATION DETECTION ============
    // Catches "how many", "count of", "which zone has the most", "zone-wise",
    // "breakdown", "funnel", "compare X vs Y", "trend of", "average ... across",
    // ageing/overdue/stuck/idle/unassigned phrasing — anything that requires
    // scanning the FULL dataset rather than a handful of similar chunks.
    const AGGREGATION_PATTERNS =
      /\b(how many|count of|count|number of|which (zone|city|team member|gm|owner)s?\s+has\s+(the\s+)?(most|highest|least|lowest|fewest)|breakdown|split of|funnel( view)?|trend of|compare\b.*\b(zones?|cities?|months?)|average\b.*\b(cost|value|ticket size|timeline|area)|zone-wise|city-wise|team member-wise|owner-wise|gm-wise|by ageing bucket|behind schedule|stuck (in|at|between)|pending (for )?more than|older than \d|idle( for)?|unassigned|no assigned owner|open (rfqs?|pos?)|overdue|awaiting (vendor|client|approval)|last (\d+ )?(months?|weeks?|days?)\b.*\b(vs|versus|previous)\b)\b/i;

    const extractAggregationIntent = (
      question: string,
    ): {
      groupBy?: "zone" | "city" | "stage" | "status" | "owner" | "leadType";
      metric: "count" | "sum_boq" | "avg_boq" | "avg_area";
      extraFilters: Record<string, any>;
    } => {
      const q = question.toLowerCase();

      let groupBy: "zone" | "city" | "stage" | "status" | "owner" | "leadType" | undefined;

      if (/zone-wise|by zone|each zone|per zone|which zone|across zones/.test(q)) {
        groupBy = "zone";
      } else if (/city-wise|by city|each city|per city|which city|across cities|city wise/.test(q)) {
        groupBy = "city";
      } else if (/by stage|each stage|current stage|funnel|stuck in|stuck at|which stage/.test(q)) {
        groupBy = "stage";
      } else if (/gm-wise|owner-wise|team member-wise|by (gm|owner)|each gm|which (gm|team member|owner)/.test(q)) {
        groupBy = "owner";
      } else if (/by status|active projects|completed projects|in progress projects/.test(q)) {
        groupBy = "status";
      }

      let metric: "count" | "sum_boq" | "avg_boq" | "avg_area" = "count";

      if (/average\b.*\b(cost|value|ticket size)|avg (cost|value)/.test(q)) {
        metric = "avg_boq";
      } else if (/total\b.*\b(cost|value)|sum of/.test(q)) {
        metric = "sum_boq";
      } else if (/average\b.*\b(area|timeline)/.test(q)) {
        metric = "avg_area";
      }

      const extraFilters: Record<string, any> = {};
      // Reuse existing single-value extractors so filters stay consistent
      // with the rest of the pipeline (stage/status keyword maps defined below).
      return { groupBy, metric, extraFilters };
    };

    const classifyRoute = (question: string): RouteType => {
      const normalizedQuestion = normalizeQuestionText(question);
      const lowerQuestion = normalizedQuestion.toLowerCase();

      // Aggregation check runs FIRST and can override flow/cost/zone signals,
      // because "zone-wise cost" or "projects stuck in design stage" are
      // counting/grouping questions, not single-project lookups.
      if (AGGREGATION_PATTERNS.test(lowerQuestion)) {
        return "aggregation";
      }

      const flowIntent = extractFlowIntent(normalizedQuestion);

      const hasFlowSignal =
        flowIntent.hasFlowKeyword ||
        /\b(stage|substage|phase|milestone|timeline|schedule|progress|flow|process|handover|execution|mobilisation|mobilization|kickoff|gfc)\b/i.test(
          normalizedQuestion,
        );

      if (hasFlowSignal) {
        return "flow_analysis";
      }

      const hasCostSignal =
        /\b(cost|budget|overrun|price|value|financial|expense|boq|lakh|lac|crore|cr|amount|estimate|estimated)\b/i.test(
          lowerQuestion,
        );

      if (hasCostSignal) {
        return "cost";
      }

      const hasZoneSignal =
        (/across|compare|analysis|breakdown|performance/i.test(lowerQuestion) &&
          (/\b(north|south|east|west)\b/i.test(normalizedQuestion) ||
            /\b(bangalore|mumbai|kolkata|gurgaon|delhi|pune|hyderabad|chennai|vadodara)\b/i.test(
              normalizedQuestion,
            ))) ||
        /\b(north|south|east|west)\b.*\b(north|south|east|west)\b/i.test(
          normalizedQuestion,
        );

      if (hasZoneSignal) {
        return "zone_analysis";
      }

      const hasSummarySignal =
        /\b(summary|overview|status|achievement|snapshot|current status|what's the status|what is the status)\b/i.test(
          lowerQuestion,
        );

      if (hasSummarySignal) {
        return "summary";
      }

      return "rag";
    };

    const resolveRoute = (question: string, subQuestions: string[] = []): RouteType => {
      const candidates = [question, ...(subQuestions || [])].filter(Boolean);
      const candidateRoutes = candidates.map((candidate) => classifyRoute(candidate));

      if (candidateRoutes.includes("aggregation")) {
        return "aggregation";
      }

      if (candidateRoutes.includes("flow_analysis")) {
        return "flow_analysis";
      }

      if (candidateRoutes.includes("cost")) {
        return "cost";
      }

      if (candidateRoutes.includes("zone_analysis")) {
        return "zone_analysis";
      }

      if (candidateRoutes.includes("summary")) {
        return "summary";
      }

      return "rag";
    };

    // ============ 2. EXTRACT FLOW INTENT ============

    const extractFlowIntent = (question: string): {
      hasFlowKeyword: boolean;
      flowType: 'phase' | 'milestone' | 'timeline' | 'progress' | 'stage' | 'general';
      keywords: string[];
    } => {
      const lowerQuestion = question.toLowerCase();

      const phaseKeywords = ['phase', 'pre-sale', 'pre sales', 'design-sale', 'design sale', 'design delivery', 'execution', 'handover', 'stage', 'substage'];
      const milestoneKeywords = ['milestone', 'gfc', 'boq', 'mobilisation', 'mobilization', 'kickoff', 'completion', 'handover'];
      const timelineKeywords = ['timeline', 'due date', 'schedule', 'duration', 'deadline', 'completion date'];
      const progressKeywords = ['progress', 'status', 'completed', 'in progress', 'pending', 'completion %', 'current status'];
      const generalFlowKeywords = ['flow', 'process', 'project flow', 'phases', 'stages', 'what are the steps', 'what is the process'];

      const collectMatches = (keywords: string[]) =>
        keywords.filter((keyword) => lowerQuestion.includes(keyword));

      const foundKeywords = [
        ...new Set([
          ...collectMatches(phaseKeywords),
          ...collectMatches(milestoneKeywords),
          ...collectMatches(timelineKeywords),
          ...collectMatches(progressKeywords),
          ...collectMatches(generalFlowKeywords),
        ]),
      ];

      let flowType: 'phase' | 'milestone' | 'timeline' | 'progress' | 'stage' | 'general' = 'general';

      if (collectMatches(phaseKeywords).length) {
        flowType = 'phase';
      } else if (collectMatches(milestoneKeywords).length) {
        flowType = 'milestone';
      } else if (collectMatches(timelineKeywords).length) {
        flowType = 'timeline';
      } else if (collectMatches(progressKeywords).length) {
        flowType = 'progress';
      } else if (collectMatches(generalFlowKeywords).length) {
        flowType = 'general';
      }

      const hasFlowKeyword = foundKeywords.length > 0;

      if (hasFlowKeyword) {
        console.log(`🗺️ Flow intent detected: type="${flowType}", keywords=[${foundKeywords.join(', ')}]`);
      }

      return {
        hasFlowKeyword,
        flowType,
        keywords: foundKeywords,
      };
    };

    // ============ 3. EXTRACT STAGE ============

    const extractStage = (question: string): string | undefined => {
      const lowerQuestion = question.toLowerCase();

      const stageMap: Record<string, string> = {
        'pre sales': 'Pre-Sales',
        'pre-sales': 'Pre-Sales',
        'presales': 'Pre-Sales',
        'pre sale': 'Pre-Sales',
        'sales': 'Pre-Sales',

        'design sales': 'Design-Sales',
        'design-sales': 'Design-Sales',
        'designsales': 'Design-Sales',
        'sales design': 'Design-Sales',

        'design delivery': 'Design Delivery',
        'design-delivery': 'Design Delivery',
        'designdelivery': 'Design Delivery',
        'delivery': 'Design Delivery',
        'design': 'Design Delivery',

        'execution': 'Execution',
        'executing': 'Execution',
        'execute': 'Execution',
        'executed': 'Execution',
        'in execution': 'Execution',
        'under execution': 'Execution',

        'handover': 'Handover',
        'hand over': 'Handover',
        'hand-over': 'Handover',
        'handed over': 'Handover',
        'completed': 'Handover',
        'completion': 'Handover',
        'finished': 'Handover',
        'done': 'Handover',
      };

      for (const [keyword, stageValue] of Object.entries(stageMap)) {
        const escapedKeyword = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const boundaryPattern = new RegExp(`\\b${escapedKeyword}\\b`, 'i');

        if (boundaryPattern.test(lowerQuestion)) {
          console.log(`📍 Stage extracted: "${keyword}" → "${stageValue}"`);
          return stageValue;
        }
      }

      return undefined;
    };

    // ============ 4. EXTRACT STATUS ============

    const extractStatus = (question: string): string | undefined => {
      const lowerQuestion = question.toLowerCase();

      const statusMap: Record<string, string> = {
        'cancelled': 'Cancelled',
        'canceled': 'Cancelled',
        'cancel': 'Cancelled',
        'terminated': 'Cancelled',
        'dropped': 'Cancelled',
        'abandoned': 'Cancelled',

        'completed': 'Completed',
        'complete': 'Completed',
        'completion': 'Completed',
        'finished': 'Completed',
        'finish': 'Completed',
        'done': 'Completed',
        'closed': 'Completed',

        'in progress': 'InProgress',
        'in-progress': 'InProgress',
        'inprogress': 'InProgress',
        'ongoing': 'InProgress',
        'active': 'InProgress',
        'running': 'InProgress',
        'under progress': 'InProgress',
        'work in progress': 'InProgress',

        'on hold': 'OnHold',
        'onhold': 'OnHold',
        'hold': 'OnHold',
        'paused': 'OnHold',
        'pause': 'OnHold',
        'suspended': 'OnHold',
        'deferred': 'OnHold',
      };

      for (const [keyword, statusValue] of Object.entries(statusMap)) {
        const escapedKeyword = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const boundaryPattern = new RegExp(`\\b${escapedKeyword}\\b`, 'i');

        if (boundaryPattern.test(lowerQuestion)) {
          console.log(`🔵 Status extracted: "${keyword}" → "${statusValue}"`);
          return statusValue;
        }
      }

      return undefined;
    };

    // ============ 5. EXTRACT COST RANGE ============

    const extractCostRange = (question: string): {
      min?: number;
      max?: number;
      operator?: string;
      hasCostKeyword?: boolean;
    } => {
      const result: {
        min?: number;
        max?: number;
        operator?: string;
        hasCostKeyword?: boolean;
      } = {};

      const lowerQuestion = question.toLowerCase();

      const costKeywords = [
        'cost',
        'boqvalue',
        'boquvalue',
        'boq',
        'budget',
        'price',
        'financial',
        'amount',
        'value',
        'overrun',
        'spend',
        'expense',
        'payment',
      ];

      const hasCostKeyword = costKeywords.some((keyword) =>
        lowerQuestion.includes(keyword)
      );

      if (!hasCostKeyword) {
        console.log(`🚫 Cost extraction skipped: no cost keyword found`);
        return result;
      }

      result.hasCostKeyword = true;

      const convertToRupees = (value: number, unit: string): number => {
        const normalizedUnit = unit.toLowerCase();

        if (normalizedUnit === 'k') {
          return value * 1000;
        }

        if (normalizedUnit === 'c' || normalizedUnit === 'crore') {
          return value * 10000000;
        }

        if (
          normalizedUnit === 'l' ||
          normalizedUnit === 'lakh' ||
          normalizedUnit === 'lac'
        ) {
          return value * 100000;
        }

        return value;
      };

      const greaterMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:greater\s+than|more\s+than|>|above|exceeds?)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (greaterMatch) {
        const numberMatch = greaterMatch[0].match(/(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);
          result.min = value;
          result.operator = 'greater_than';

          console.log(`💰 Cost extracted (cost > pattern): ${numberMatch[1]} ${unit} = ₹${value.toLocaleString('en-IN')}`);
        }
      }

      const lessMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:less\s+than|under|<|below|upto|up\s+to)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (lessMatch) {
        const numberMatch = lessMatch[0].match(/(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);
          result.max = value;
          result.operator = 'less_than';

          console.log(`💰 Cost extracted (cost < pattern): ${numberMatch[1]} ${unit} = ₹${value.toLocaleString('en-IN')}`);
        }
      }

      const betweenMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:between|from|through)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (betweenMatch) {
        const min = parseFloat(betweenMatch[1]);
        const max = parseFloat(betweenMatch[2]);
        const unit = betweenMatch[3] || 'lakh';

        result.min = convertToRupees(min, unit);
        result.max = convertToRupees(max, unit);
        result.operator = 'between';

        console.log(
          `💰 Cost extracted (cost between): ${min}-${max} ${unit} = ₹${result.min.toLocaleString('en-IN')}-₹${result.max.toLocaleString('en-IN')}`
        );
      }

      const currencyMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+[^\d]*₹\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (currencyMatch) {
        const numberMatch = currencyMatch[0].match(/₹\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);

          if (!result.min && !result.max) {
            result.min = value;
            console.log(`💰 Cost extracted (currency): ₹${value.toLocaleString('en-IN')}`);
          }
        }
      }

      if (result.min || result.max || result.operator) {
        console.log('✅ Final cost range:', {
          min: result.min ? `₹${result.min.toLocaleString('en-IN')}` : undefined,
          max: result.max ? `₹${result.max.toLocaleString('en-IN')}` : undefined,
          operator: result.operator,
          hasCostKeyword: result.hasCostKeyword,
        });
      }

      return result;
    };

    const extractMappedValue = (
      question: string,
      map: Record<string, string>,
    ): string[] => {
      const lower = question.toLowerCase();

      return [
        ...new Set(
          Object.entries(map)
            .filter(([alias]) => lower.includes(alias))
            .map(([, value]) => value),
        ),
      ];
    };

    // ============ 6. EXTRACT AREA RANGE ============

    const extractAreaRange = (question: string): { min?: number; max?: number; unit?: string; operator?: string } => {
      const result: { min?: number; max?: number; unit?: string; operator?: string } = {};

      const greaterMatch = question.match(/(?:greater\s+than|more\s+than|>|above)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (greaterMatch) {
        let value = parseFloat(greaterMatch[1]);
        const unit = greaterMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.min = value;
        result.unit = "sqft";
        result.operator = "greater_than";
      }

      const lessMatch = question.match(/(?:less\s+than|under|<|below)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (lessMatch) {
        let value = parseFloat(lessMatch[1]);
        const unit = lessMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.max = value;
        result.unit = "sqft";
        result.operator = "less_than";
      }

      const betweenMatch = question.match(/between\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (betweenMatch) {
        const unit = betweenMatch[3]?.toLowerCase().replace(/\./g, '') || "sqft";
        let min = parseFloat(betweenMatch[1]);
        let max = parseFloat(betweenMatch[2]);

        if (unit === 'sqm') {
          min = min * 10.764;
          max = max * 10.764;
        }

        result.min = min;
        result.max = max;
        result.unit = "sqft";
        result.operator = "between";
      }

      const aboveMatch = question.match(/above\s*(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (aboveMatch) {
        let value = parseFloat(aboveMatch[1]);
        const unit = aboveMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.min = value;
        result.unit = "sqft";
        result.operator = "above";
      }

      if (Object.keys(result).length > 0) {
        console.log("📐 Extracted area range:", result);
      }

      return result;
    };

    // ============ 7. INTENT DETECTION (UPDATED) ============

    const detectIntent = (question: string): any => {
      const projectIdMatch = question.match(/project\s+(\d+)/i);
      const projectId = projectIdMatch?.[1];

      const leadTypeMap: Record<string, string> = {
        'new office': 'New Office',
        'newoffice': 'New Office',
        'new': 'New Office',
        'office renovation': 'Office Renovation',
        'office-renovation': 'Office Renovation',
        'officereno': 'Office Renovation',
        'renovation': 'Renovation',
        'renovate': 'Renovation',
        'reno': 'Renovation',
      };

      const ownerMap: Record<string, string> = {
        'enterprise team': 'Enterprise Team',
        'enterprise': 'Enterprise Team',
        'product team': 'Product Team',
        'product': 'Product Team',
        'random owner 2': 'Random Owner 2',
        'random owner2': 'Random Owner 2',
        'owner 2': 'Random Owner 2',
        'skv': 'SKV',
        'smb team': 'SMB Team',
        'smb': 'SMB Team',
        'team b': 'Team B',
        'teamb': 'Team B',
      };

      const LEAD_ZONES = {
        Bangalore: "South",
        Kolkata: "East",
        Mumbai: "West",
        Gurgaon: "North",
        Unassigned: "Unassigned",
      } as const;

      const REGION_TO_ZONES = Object.entries(LEAD_ZONES).reduce(
        (acc, [zoneName, region]) => {
          const key = region.toLowerCase();

          if (!acc[key]) {
            acc[key] = [];
          }

          acc[key].push(zoneName);

          return acc;
        },
        {} as Record<string, string[]>,
      );

      const zoneMatches = question.match(
        /\b(north|south|east|west|north-?east|north-?west|south-?east|south-?west)\b/gi,
      );

      let zones: string[] | undefined;

      if (zoneMatches) {
        const regions = Array.from(new Set(zoneMatches.map((z) => z.toLowerCase())));

        const expandedRegions = regions
          .flatMap((region) => {
            switch (region) {
              case "north-east":
              case "northeast":
                return ["north", "east"];

              case "north-west":
              case "northwest":
                return ["north", "west"];

              case "south-east":
              case "southeast":
                return ["south", "east"];

              case "south-west":
              case "southwest":
                return ["south", "west"];

              default:
                return [region];
            }
          });

        zones = [
          ...new Set(
            expandedRegions.flatMap(
              (region) => REGION_TO_ZONES[region] ?? [],
            ),
          ),
        ];
      }

      console.log("Detected zones:", zones);

      const cityMatches = question.match(
        /(vadodara|bangalore|delhi|mumbai|pune|hyderabad|chennai|kolkata)/gi
      );
      const cities = cityMatches
        ? Array.from(new Set(cityMatches.map((c) => c.toLowerCase())))
        : undefined;

      const stage = extractStage(question);
      const status = extractStatus(question);

      const costRange = extractCostRange(question);
      const areaRange = extractAreaRange(question);
      const leadTypes = extractMappedValue(question, leadTypeMap);
      const owners = extractMappedValue(question, ownerMap);

      const flowIntent = extractFlowIntent(question);
      const type = classifyRoute(question);
      const aggregationIntent = type === "aggregation" ? extractAggregationIntent(question) : undefined;

      if (type === "flow_analysis") {
        console.log(`🗺️ Routing to flow_analysis agent`);
      }

      if (type === "aggregation") {
        console.log(`📊 Routing to aggregation agent`, aggregationIntent);
      }

      if (zones?.length) {
        for (const [zone, region] of Object.entries(LEAD_ZONES)) {
          const regex = new RegExp(`\\b${region}\\b`, "gi");
          question = question.replace(regex, zone);
        }
      }

      return {
        type,
        projectId,
        zones,
        cities,
        stage,
        status,
        leadTypes,
        owners,
        costRange,
        areaRange,
        flowIntent,
        aggregationIntent,
        limit:
          type === "zone_analysis"
            ? 25
            : type === "flow_analysis"
              ? 20
              : type === "aggregation"
                ? 0 // aggregation doesn't use vector-search limit; it scrolls everything
                : type === "rag"
                  ? 7
                  : 15,
        question,
      };
    };

    // ============ 8. ORGANIZE CHUNKS ============

    const organizeChunksByType = (chunks: any[]): Record<string, any[]> => {
      const organized: Record<string, any[]> = {
        details: [],
        flow: [],
        financial: [],
      };

      chunks.forEach((chunk) => {
        const docType = chunk.payload?.docType || "details";
        if (!organized[docType]) {
          organized[docType] = [];
        }
        organized[docType].push(chunk);
      });

      return organized;
    };

    // ============ 9. ZONE ANALYSIS AGGREGATION ============

    const aggregateByZone = (chunks: any[]): Record<string, any[]> => {
      const byZone: Record<string, any[]> = {};

      chunks.forEach((chunk) => {
        const zone = chunk.payload?.zone || "Unknown";
        if (!byZone[zone]) {
          byZone[zone] = [];
        }
        byZone[zone].push(chunk);
      });

      return byZone;
    };

    const aggregateByCity = (chunks: any[]): Record<string, any[]> => {
      const byCity: Record<string, any[]> = {};

      chunks.forEach((chunk) => {
        const city = chunk.payload?.city || "Unknown";
        if (!byCity[city]) {
          byCity[city] = [];
        }
        byCity[city].push(chunk);
      });

      return byCity;
    };

    // ============ 9b. NEW: FULL-DATASET AGGREGATION VIA QDRANT SCROLL ============
    // Unlike vector search (top-k similarity), this paginates through EVERY
    // point matching the filters, so counts/breakdowns are exact — not an
    // artifact of whatever the embedding search happened to retrieve.

    const runAggregation = async (
      intent: any,
      embedding: number[]
    ): Promise<{
      total: number;
      groups: Record<string, { count: number; boqSum: number; areaSum: number }>;
    }> => {
      const mustFilters: any[] = [
        {
          key: "docType",
          match: { value: "details" }, // one "details" chunk per project avoids double counting
        },
      ];

      if (intent.zones && intent.zones.length > 0) {
        mustFilters.push({
          should: intent.zones.map((zone: string) => ({
            key: "zone",
            match: { value: zone },
          })),
        });
      }

      if (intent.cities && intent.cities.length > 0) {
        mustFilters.push({
          should: intent.cities.map((city: string) => ({
            key: "city",
            match: { value: city },
          })),
        });
      }

      if (intent.stage) {
        mustFilters.push({ key: "stage", match: { value: intent.stage } });
      }

      if (intent.status) {
        mustFilters.push({ key: "status", match: { value: intent.status } });
      }

      if (intent.owners?.length) {
        mustFilters.push({
          should: intent.owners.map((owner: string) => ({
            key: "metadata.owner",
            match: { value: owner },
          })),
        });
      }

      console.log("📊 Aggregation filters:", JSON.stringify(mustFilters));

      const allPoints: any[] = [];
      let offset: number | undefined | null = undefined;

      do {
        const page: any = await this.qdrant.scrollAll("collection_gemini", {
          filter: { must: mustFilters },
          limit: 250,
          offset: offset ?? undefined,
          with_payload: true,
          with_vector: false,
        });

        const points = page?.points || page?.result?.points || [];
        allPoints.push(...points);
        offset = page?.next_page_offset ?? page?.result?.next_page_offset ?? null;
      } while (offset);

      console.log(`📊 Aggregation scrolled ${allPoints.length} total records`);

      const groupByField = intent.aggregationIntent?.groupBy as
        | "zone"
        | "city"
        | "stage"
        | "status"
        | "owner"
        | "leadType"
        | undefined;

      if (!groupByField) {
        const totalBoq = allPoints.reduce((sum, p) => sum + (p.payload?.boqValue || 0), 0);
        return {
          total: allPoints.length,
          groups: { "All Projects": { count: allPoints.length, boqSum: totalBoq, areaSum: 0 } },
        };
      }

      const groups: Record<string, { count: number; boqSum: number; areaSum: number }> = {};

      allPoints.forEach((point) => {
        let key: string;
        if (groupByField === "owner") {
          key = point.payload?.metadata?.owner || point.payload?.owner || "Unassigned";
        } else {
          key = point.payload?.[groupByField] || "Unknown";
        }

        if (!groups[key]) {
          groups[key] = { count: 0, boqSum: 0, areaSum: 0 };
        }

        groups[key].count += 1;
        groups[key].boqSum += point.payload?.boqValue || 0;
        groups[key].areaSum += point.payload?.area || 0;
      });

      return { total: allPoints.length, groups };
    };

    // ============ 10. FORMAT FUNCTIONS ============

    const formatCurrency = (value: number): string => {
      if (value >= 10000000) {
        return `₹${(value / 10000000).toFixed(1)} Cr`;
      } else if (value >= 100000) {
        return `₹${(value / 100000).toFixed(1)} L`;
      } else {
        return `₹${value.toLocaleString("en-IN")}`;
      }
    };

    // NEW: Format aggregation context — exact numbers only, LLM just phrases them.
    const formatAggregationContext = (
      result: { total: number; groups: Record<string, { count: number; boqSum: number; areaSum: number }> },
      aggregationIntent: { groupBy?: string; metric: string } | undefined,
    ): string => {
      const sections: string[] = [];

      sections.push("AGGREGATED PROJECT DATA (exact counts from full dataset scan)");
      sections.push("=".repeat(80));
      sections.push("");
      sections.push(`Total matching projects: ${result.total}`);
      sections.push("");

      const groupBy = aggregationIntent?.groupBy;
      const metric = aggregationIntent?.metric || "count";

      if (groupBy && Object.keys(result.groups).length > 0) {
        sections.push(`BREAKDOWN BY ${groupBy.toUpperCase()}:`);
        sections.push("-".repeat(80));

        Object.entries(result.groups)
          .sort((a, b) => b[1].count - a[1].count)
          .forEach(([key, stats]) => {
            let line = `${key}: ${stats.count} projects`;

            if (metric === "sum_boq") {
              line += `, total value ${formatCurrency(stats.boqSum)}`;
            } else if (metric === "avg_boq" && stats.count > 0) {
              line += `, avg value ${formatCurrency(stats.boqSum / stats.count)}`;
            } else if (metric === "avg_area" && stats.count > 0) {
              line += `, avg area ${(stats.areaSum / stats.count).toFixed(0)} sqft`;
            }

            sections.push(line);
          });
      } else {
        Object.entries(result.groups).forEach(([key, stats]) => {
          sections.push(`${key}: ${stats.count} projects, total value ${formatCurrency(stats.boqSum)}`);
        });
      }

      sections.push("");
      sections.push(
        "IMPORTANT: These numbers come from an exhaustive scan of the matching dataset, not a sample. Report them exactly — do not round differently or estimate.",
      );

      return sections.join("\n");
    };

    // NEW: Format flow context
    const formatFlowContext = (
      chunksByType: Record<string, any[]>,
      flowIntent: any
    ): string => {
      const sections: string[] = [];

      sections.push("PROJECT FLOW AND PHASES");
      sections.push("=".repeat(80));
      sections.push("");
      interface Stage {
        phase: string;
        stageNo: number;
        name: string;
        status: string;
        due: string;
        duration: string;
        roles: string[];
      }

      function parseStages(flowText: string): Stage[] {
        const stages: Stage[] = [];

        let currentPhase = "";

        const lines = flowText.split("\n");

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();

          if (
            line.startsWith("PROJECT FLOW") ||
            line.startsWith("PHASE SUMMARY") ||
            line.startsWith("DETAILED STAGES") ||
            line.startsWith("===") ||
            line.startsWith("---") ||
            line.startsWith("Total Stages")
          ) {
            continue;
          }

          if (/^[A-Za-z\s-]+:$/.test(line)) {
            currentPhase = line.replace(":", "").trim();
            continue;
          }

          const stageMatch = line.match(/^(\d+)\.\s+(.*)$/);

          if (!stageMatch) continue;

          const stage: Stage = {
            phase: currentPhase,
            stageNo: Number(stageMatch[1]),
            name: stageMatch[2].trim(),
            status: "",
            due: "",
            duration: "",
            roles: [],
          };

          while (++i < lines.length) {
            const next = lines[i].trim();

            if (/^\d+\./.test(next)) {
              i--;
              break;
            }

            if (/^[A-Za-z\s-]+:$/.test(next)) {
              i--;
              break;
            }

            if (next.startsWith("Status:")) {
              stage.status = next.replace("Status:", "").trim();
            }

            if (next.startsWith("Due:")) {
              const match = next.match(/Due:\s*(.*?)\s*\|\s*Duration:\s*(.*)/);

              if (match) {
                stage.due = match[1];
                stage.duration = match[2];
              }
            }

            if (next.startsWith("Roles:")) {
              stage.roles = next
                .replace("Roles:", "")
                .split(",")
                .map((x) => x.trim());
            }
          }

          stages.push(stage);
        }

        return stages;
      }

      if (chunksByType.flow?.length) {
        const flowData = chunksByType.flow[0];
        const flowText = flowData.payload?.text || "";

        const stages = parseStages(flowText);

        switch (flowIntent.flowType) {
          case "phase": {
            sections.push("PROJECT PHASES");
            sections.push("-".repeat(80));

            const phaseMap = new Map<string, Stage[]>();

            stages.forEach((stage) => {
              if (!phaseMap.has(stage.phase)) {
                phaseMap.set(stage.phase, []);
              }

              phaseMap.get(stage.phase)!.push(stage);
            });

            phaseMap.forEach((phaseStages, phase) => {
              sections.push("");
              sections.push(`${phase.toUpperCase()}`);

              phaseStages.forEach((stage) => {
                const icon =
                  stage.status === "COMPLETED"
                    ? "✔"
                    : stage.status === "IN_PROGRESS"
                      ? "⏳"
                      : "○";

                sections.push(
                  `  ${icon} ${stage.stageNo}. ${stage.name} (${stage.status})`
                );
              });
            });

            break;
          }

          case "timeline": {
            sections.push("PROJECT TIMELINE");
            sections.push("-".repeat(80));

            stages.forEach((stage) => {
              sections.push(
                `${stage.due.padEnd(12)} | ${stage.name} (${stage.phase})`
              );
            });

            break;
          }

          case "milestone": {
            sections.push("PROJECT MILESTONES");
            sections.push("-".repeat(80));

            const milestones = stages.filter((stage) =>
              stage.name.toLowerCase().includes("milestone")
            );

            if (!milestones.length) {
              sections.push("No milestones found.");
            } else {
              milestones.forEach((stage) => {
                sections.push(
                  `${stage.name}
                  Status: ${stage.status}
                  Due: ${stage.due}
                  Duration: ${stage.duration}
                  `
                );
              });
            }

            break;
          }

          case "progress": {
            sections.push("PROJECT PROGRESS");
            sections.push("-".repeat(80));

            let completed = 0;
            let inProgress = 0;
            let pending = 0;

            stages.forEach((stage) => {
              const icon =
                stage.status === "COMPLETED"
                  ? "✔"
                  : stage.status === "IN_PROGRESS"
                    ? "⏳"
                    : "○";

              sections.push(
                `${icon} ${stage.name} (${stage.phase}) - ${stage.status}`
              );

              switch (stage.status) {
                case "COMPLETED":
                  completed++;
                  break;

                case "IN_PROGRESS":
                  inProgress++;
                  break;

                default:
                  pending++;
              }
            });

            sections.push("");
            sections.push("SUMMARY");
            sections.push("-".repeat(40));
            sections.push(`Completed : ${completed}`);
            sections.push(`In Progress : ${inProgress}`);
            sections.push(`Pending : ${pending}`);
            sections.push(`Total : ${stages.length}`);

            sections.push(
              `Completion : ${((completed / stages.length) * 100).toFixed(1)}%`
            );

            break;
          }

          default: {
            sections.push("COMPLETE PROJECT FLOW");
            sections.push("-".repeat(80));
            sections.push(flowText);
          }
        }

        sections.push("");
      }

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT TEAM & ROLES");
        sections.push("-".repeat(80));
        const detailsData = chunksByType.details[0];
        const detailsText = detailsData.payload?.text || "";
        const roleMatches = detailsText.match(/Role[^\n]*|Manager[^\n]*/g) || [];
        if (roleMatches.length > 0) {
          sections.push(roleMatches.slice(0, 10).join("\n"));
        }
      }

      return sections.join("\n");
    };

    const formatAreaRangeContext = (
      chunksByType: Record<string, any[]>,
      areaRange: { min?: number; max?: number; unit?: string }
    ): string => {
      const sections: string[] = [];

      sections.push("AREA ANALYSIS");
      sections.push("=".repeat(80));
      sections.push("");

      if (areaRange.min && areaRange.max) {
        sections.push(`Filter: ${areaRange.min.toFixed(0)} - ${areaRange.max.toFixed(0)} sqft`);
      } else if (areaRange.min) {
        sections.push(`Filter: Area > ${areaRange.min.toFixed(0)} sqft`);
      } else if (areaRange.max) {
        sections.push(`Filter: Area < ${areaRange.max.toFixed(0)} sqft`);
      }
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("LARGE-AREA PROJECTS:");
        sections.push("-".repeat(80));

        const sorted = [...chunksByType.details].sort((a, b) => {
          const aArea = a.payload?.area || 0;
          const bArea = b.payload?.area || 0;
          return bArea - aArea;
        });

        sorted.forEach((chunk, idx) => {
          const area = chunk.payload?.area || 0;
          const projectCode = chunk.payload?.projectCode || "Unknown";
          const zone = chunk.payload?.zone || "Unknown";

          sections.push(`\n[Project ${idx + 1}] ${projectCode} (${zone})`);
          sections.push(`Area: ${area.toFixed(0)} sqft`);
          sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    const formatCostRangeContext = (
      chunksByType: Record<string, any[]>,
      costRange: { min?: number; max?: number; operator?: string }
    ): string => {
      const sections: string[] = [];

      sections.push("COST RANGE ANALYSIS");
      sections.push("=".repeat(80));
      sections.push("");

      if (costRange.min && costRange.max) {
        sections.push(`Filter: ${formatCurrency(costRange.min)} - ${formatCurrency(costRange.max)}`);
      } else if (costRange.min) {
        sections.push(`Filter: Cost > ${formatCurrency(costRange.min)}`);
      } else if (costRange.max) {
        sections.push(`Filter: Cost < ${formatCurrency(costRange.max)}`);
      }
      sections.push("");

      if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("HIGH-VALUE PROJECTS:");
        sections.push("-".repeat(80));

        const sorted = [...chunksByType.financial].sort((a, b) => {
          const aValue = a.payload?.boqValue || 0;
          const bValue = b.payload?.boqValue || 0;
          return bValue - aValue;
        });

        sorted.forEach((chunk, idx) => {
          const value = chunk.payload?.boqValue || 0;
          const projectCode = chunk.payload?.projectCode || "Unknown";
          const zone = chunk.payload?.zone || "Unknown";

          sections.push(`\n[Project ${idx + 1}] ${projectCode} (${zone})`);
          sections.push(`Cost: ${formatCurrency(value)}`);
          sections.push(`area: ${chunk.payload?.area ? chunk.payload.area.toFixed(0) + " sqft" : "N/A"}`);
          sections.push(chunk.payload?.text || "");

        });
      }

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("\nPROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          const text = chunk.payload?.text || "";
          const boqValue = chunk.payload?.boqValue || 0;
          const relevantLines = text
            .split("\n")
            .filter((line: string) =>
              /project|budget|cost|estimated|actual|value|area|scope/i.test(line)
            );
          if (relevantLines.length > 0) {
            sections.push(relevantLines.join("\n"));
            sections.push(`Cost of ${chunk.payload?.projectCode || "Unknown"}: ${formatCurrency(boqValue)}`);
          }
        });
      }

      return sections.join("\n");
    };

    const formatCostAnalysisContext = (
      chunksByType: Record<string, any[]>,
      costRange?: { min?: number; max?: number }
    ): string => {
      if (costRange && (costRange.min !== undefined || costRange.max !== undefined)) {
        return formatCostRangeContext(chunksByType, costRange);
      }

      const sections: string[] = [];

      sections.push("FINANCIAL ANALYSIS CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("COST DATA:");
        sections.push("-".repeat(80));
        chunksByType.financial.forEach((chunk, idx) => {
          sections.push(`[Document ${idx + 1}]`);
          sections.push(chunk.payload?.text || "");
          sections.push("");
        });
      }

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          const text = chunk.payload?.text || "";
          const relevantLines = text
            .split("\n")
            .filter((line: string) =>
              /project|budget|cost|estimated|actual|value|area|scope/i.test(line)
            );
          if (relevantLines.length > 0) {
            sections.push(relevantLines.join("\n"));
          }
        });
      }

      return sections.join("\n");
    };

    const formatSummaryContext = (
      chunksByType: Record<string, any[]>
    ): string => {
      const sections: string[] = [];

      sections.push("PROJECT SUMMARY CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT INFORMATION:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
        sections.push("");
      }

      if (chunksByType.flow && chunksByType.flow.length > 0) {
        sections.push("PROJECT STATUS:");
        sections.push("-".repeat(80));
        chunksByType.flow.forEach((chunk) => {
          // sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    const formatZoneAnalysisContext = (
      byZone: Record<string, any[]>,
      byCity: Record<string, any[]>,
      intent: any
    ): string => {
      const sections: string[] = [];

      sections.push("ZONE/REGION ANALYSIS CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (intent.zones && intent.zones.length > 0) {
        sections.push("DATA BY ZONE:");
        sections.push("-".repeat(80));

        intent.zones.forEach((zone: string) => {
          const zoneChunks = byZone[zone] || [];
          if (zoneChunks.length > 0) {
            sections.push(`\n📍 ZONE: ${zone.toUpperCase()}`);
            sections.push(`   Projects: ${zoneChunks.length}`);

            const projects = new Set(zoneChunks.map((c) => c.payload?.projectId));
            sections.push(`   Unique Projects: ${projects.size}`);

            zoneChunks.slice(0, 2).forEach((chunk) => {
              const text = chunk.payload?.text || "";
              const projectName = chunk.payload?.metadata?.projectName || "Unknown Project";
              const customerName = chunk.payload?.metadata?.customerName || "Unknown Customer";
              let summary = `Project: ${projectName}, Customer: ${customerName}, cost: ${chunk.payload?.boqValue}`;
              summary = text.split("\n").slice(0, 3).join("\n");
              sections.push(`   ${summary}`);
            });
          }
        });
        sections.push("");
      }

      if (intent.cities && intent.cities.length > 0) {
        sections.push("DATA BY CITY:");
        sections.push("-".repeat(80));

        intent.cities.forEach((city: string) => {
          const cityChunks = byCity[city] || [];
          if (cityChunks.length > 0) {
            sections.push(`\n🏙️ CITY: ${city.toUpperCase()}`);

            const projects = new Set(cityChunks.map((c) => c.payload?.projectId));
            sections.push(`   Unique Projects: ${projects.size}`);

            const totalCost = cityChunks.reduce((sum, c) => {
              return sum + (c.payload?.boqValue || 0);
            }, 0);
            sections.push(`   Total Value: ${formatCurrency(totalCost)}`);
          }
        });
        sections.push("");
      }

      sections.push("AGGREGATED METRICS:");
      sections.push("-".repeat(80));

      const allChunks = Object.values(byZone).flat();
      const uniqueProjects = new Set(allChunks.map((c) => c.payload?.projectId)).size;
      const totalValue = allChunks.reduce(
        (sum, c) => sum + (c.payload?.boqValue || 0),
        0
      );
      const completedProjects = allChunks.filter(
        (c) => c.payload?.status === "Completed"
      ).length;

      sections.push(`Total Projects Analyzed: ${uniqueProjects}`);
      sections.push(`Total Value: ${formatCurrency(totalValue)}`);
      sections.push(`Completed: ${completedProjects}`);
      sections.push(`In Progress: ${allChunks.length - completedProjects}`);

      return sections.join("\n");
    };

    const formatGeneralContext = (
      chunksByType: Record<string, any[]>
    ): string => {
      const sections: string[] = [];

      sections.push("PROJECT CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
        sections.push("");
      }

      if (chunksByType.flow && chunksByType.flow.length > 0) {
        sections.push("PROJECT FLOW:");
        sections.push("-".repeat(80));
        chunksByType.flow.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
      }

      if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("FINANCIAL INFO:");
        sections.push("-".repeat(80));
        chunksByType.financial.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    // ============ 11. GET SYSTEM PROMPT ============

    const getSystemPrompt = (intentType: string): string => {
      const basePrompt = `You are a helpful AI assistant analyzing project data.
        Answer the user's question based ONLY on the provided context.
        Be specific, data-driven, and actionable.
        If information is not in the context, say so clearly. should be like conversational`;

      const prompts: Record<string, string> = {
        flow_analysis: `${basePrompt}
 
        For project flow analysis, provide:
        1. Current phase and stage of the project
        2. Completed milestones and upcoming ones
        3. Timeline and key dates
        4. Current progress percentage
        5. Next steps and critical path items
        6. Any delays or blockers if visible`,

        cost_analysis: `${basePrompt}
 
        For cost analysis, provide:
        1. Budget vs Actual comparison
        2. Cost overrun/savings percentage
        3. Key cost drivers
        4. Recommendations for optimization`,

        summary: `${basePrompt}
 
        For project summary, provide:
        1. Project overview and objectives
        2. Current status and progress
        3. Key achievements
        4. Outstanding issues
        5. Next steps`,

        zone_analysis: `${basePrompt}
 
        For zone/region analysis, provide:
        1. Summary of each zone/city analyzed
        2. Total project count and value
        3. Performance comparison across zones
        4. Top performers and underperformers
        5. Key insights and recommendations`,

        aggregation: `${basePrompt}
 
        The context contains EXACT counts and totals computed from a full scan of the dataset —
        not a sample. Report these numbers precisely as given. Do not estimate, round differently,
        or hedge on the numbers themselves. You may add brief interpretive commentary (e.g. which
        group is highest/lowest) but the figures must match the context exactly.`,
      };

      return prompts[intentType] || basePrompt;
    };

    // ============ 12. NODE FUNCTIONS ============

    const normalizeQuery = async (state: GraphState) => {
      const normalizedQuestion =
        await this.normalizeSemanticQuestion(state.question);
      const subQuestions = splitCompoundQuestions(state.question);
      return {
        normalizedQuestion,
        subQuestions: subQuestions.length > 0 ? subQuestions : [state.question],
      };
    };

    const generateEmbedding = async (state: GraphState) => {
      const embedding = await this.generateGeminiEmbedding(
        state.normalizedQuestion!
      );
      return { embedding };
    };

    const retrieveContext = async (state: GraphState) => {
      try {
        const questionsToProcess = (state.subQuestions && state.subQuestions.length > 0
          ? state.subQuestions
          : [state.question]).filter(Boolean);

        const buildContextForQuestion = async (question: string) => {
          const intent = detectIntent(question);
          console.log("🔍 Intent detected:", intent);

          // ===== AGGREGATION: bypass vector search entirely, scroll full dataset =====
          if (intent.type === "aggregation") {
            const aggResult = await runAggregation(intent, state.embedding);
            const formattedContext = formatAggregationContext(aggResult, intent.aggregationIntent);

            console.log("📝 Aggregation context generated", formattedContext);

            return {
              context: formattedContext,
              metadata: {
                documentsFound: aggResult.total,
                intent,
                aggregation: true,
              },
            };
          }

          const mustFilters: any[] = [];

          if (intent.projectId) {
            mustFilters.push({
              key: "projectCode",
              match: { value: +intent.projectId },
            });
          }

          if (intent.zones && intent.zones.length > 0) {
            mustFilters.push({
              should: intent.zones.map((zone: string) => ({
                key: "zone",
                match: { value: zone },
              })),
            });
          }

          if (intent.owners?.length) {
            mustFilters.push({
              should: intent.owners.map((owner: string) => ({
                key: "metadata.owner",
                match: { value: owner },
              })),
            });
          }

          if (intent.leadTypes?.length) {
            mustFilters.push({
              should: intent.leadTypes.map((leadType: string) => ({
                key: "metadata.leadType",
                match: { value: leadType },
              })),
            });
          }

          if (intent.cities && intent.cities.length > 0) {
            mustFilters.push({
              should: intent.cities.map((city: string) => ({
                key: "city",
                match: { value: city },
              })),
            });
          }

          if (intent.stage) {
            mustFilters.push({
              key: 'stage',
              match: { value: intent.stage },
            });
            console.log(`📍 Stage filter applied: ${intent.stage}`);
          }

          if (intent.status) {
            mustFilters.push({
              key: 'status',
              match: { value: intent.status },
            });
            console.log(`🔵 Status filter applied: ${intent.status}`);
          }

          if (intent.costRange && (intent.costRange.min !== undefined || intent.costRange.max !== undefined)) {
            const rangeFilter: any = {
              key: "boqValue",
              range: {},
            };

            if (intent.costRange.min !== undefined) {
              rangeFilter.range.gte = intent.costRange.min;
            }

            if (intent.costRange.max !== undefined) {
              rangeFilter.range.lte = intent.costRange.max;
            }

            mustFilters.push(rangeFilter);
            console.log("💰 Cost range filter applied:", rangeFilter);
          }

          if (intent.areaRange && (intent.areaRange.min !== undefined || intent.areaRange.max !== undefined)) {
            const areaFilter: any = {
              key: "area",
              range: {},
            };
            if (intent.areaRange.min !== undefined) {
              areaFilter.range.gte = intent.areaRange.min;
            }
            if (intent.areaRange.max !== undefined) {
              areaFilter.range.lte = intent.areaRange.max;
            }
            mustFilters.push(areaFilter);
            console.log("📐 Area range filter applied:", areaFilter);
          }


          // const allowedDocTypes = ["details", "flow", "financial"];
          const allowedDocTypes = ["project-flow-phase", "project"];
          mustFilters.push({
            should: allowedDocTypes.map((docType) => ({
              key: "docType",
              match: { value: docType },
            })),
          });

          console.log("🔽 Qdrant filters:", JSON.stringify(mustFilters));

          const searchDocuments = async (filters: any[], limit: number, embedding: number[]) =>
            this.qdrant.search(
              "collection_gemini",
              {
                vector: embedding,
                limit,
                filter: filters.length > 0 ? { must: filters } : undefined,
                score_threshold: 0.5,
              }
            );

          const embeddingToUse = questionsToProcess.length > 1
            ? await this.generateGeminiEmbedding(question)
            : state.embedding;

          let qdrantResults = await searchDocuments(mustFilters, intent.limit, embeddingToUse);

          console.log(`📦 Retrieved ${qdrantResults.length} documents`);

          if ((!qdrantResults || qdrantResults.length === 0) && mustFilters.length > 0) {
            console.log("⚠️ No results with filters, retrying with a broader fallback search");
            qdrantResults = await searchDocuments([], Math.max(intent.limit, 10), embeddingToUse);
            console.log(`📦 Fallback retrieved ${qdrantResults.length} documents`);
          }

          if (!qdrantResults || qdrantResults.length === 0) {
            return {
              context: "No projects found matching your criteria.",
              metadata: { documentsFound: 0, intent },
            };
          }

          const chunksByType = organizeChunksByType(qdrantResults);
          let formattedContext: string;

          if (intent.type === "flow_analysis") {
            formattedContext = formatFlowContext(chunksByType, intent.flowIntent);
          } else if (intent.type === "zone_analysis") {
            const byZone = aggregateByZone(qdrantResults);
            const byCity = aggregateByCity(qdrantResults);
            formattedContext = formatZoneAnalysisContext(byZone, byCity, intent);
          } else if (intent.type === "cost") {
            formattedContext = formatCostAnalysisContext(chunksByType, intent.costRange);
          } else if (intent.type === "summary") {
            formattedContext = formatSummaryContext(chunksByType);
          } else if (/area|sqft|sqm|space|size/i.test(question)) {
            formattedContext = formatAreaRangeContext(chunksByType, intent.areaRange);
          } else {
            formattedContext = formatGeneralContext(chunksByType);
          }

          console.log("📝 Formatted context generated", formattedContext);

          return {
            context: formattedContext,
            metadata: {
              documentsFound: qdrantResults.length,
              intent,
              chunksByType: Object.keys(chunksByType),
            },
          };
        };

        const retrievalResults = [] as Array<{ question: string; context: string; metadata: any }>;

        for (const question of questionsToProcess) {
          try {
            const result = await buildContextForQuestion(question);
            retrievalResults.push({
              question,
              context: result.context,
              metadata: result.metadata,
            });
          } catch (error) {
            console.error(`❌ Failed to retrieve context for sub-question: ${question}`, error);
            retrievalResults.push({
              question,
              context: "Unable to retrieve context for this sub-question.",
              metadata: { error: String(error) },
            });
          }
        }

        const combinedContext = retrievalResults
          .map((result, index) => {
            const heading = questionsToProcess.length > 1
              ? `\n=== SUB-QUESTION ${index + 1} ===\n${result.question}`
              : "";
            return `${heading}\n${result.context}`.trim();
          })
          .filter(Boolean)
          .join("\n\n");

        const combinedMetadata = {
          documentsFound: retrievalResults.reduce((sum, result) => sum + (result.metadata?.documentsFound || 0), 0),
          subQuestions: questionsToProcess,
          details: retrievalResults.map((result) => ({
            question: result.question,
            documentsFound: result.metadata?.documentsFound || 0,
            intent: result.metadata?.intent,
          })),
        };

        return {
          context: combinedContext || "No projects found matching your criteria.",
          question: questionsToProcess.length > 1
            ? `Please answer each of the following questions separately and clearly.\n${questionsToProcess.map((q, index) => `${index + 1}. ${q}`).join("\n")}`
            : state.question,
          metadata: combinedMetadata,
        };
      } catch (error) {
        console.error("❌ Error retrieving context:", error);
        return {
          context: "Error retrieving context. Please try again.",
          metadata: { error: String(error) },
        };
      }
    };

    const supervisorAgent = async (state: GraphState) => {
      const heuristicRoute = resolveRoute(state.question, state.subQuestions || []);

      const heuristicTriggerPattern =
        /phase|stage|substage|milestone|timeline|schedule|progress|cost|budget|zone|compare|summary|overview|status|how many|count of|which (zone|city|gm|team member|owner)|breakdown|funnel|zone-wise|city-wise|overdue|ageing|stuck|idle|unassigned/i;

      if (heuristicRoute !== "rag" || heuristicTriggerPattern.test(state.question)) {
        console.log(`🧭 Supervisor routed heuristically to: ${heuristicRoute}`);
        return {
          route: heuristicRoute,
        };
      }

      const response = await this.model.invoke([
        new SystemMessage(`You are a routing classifier. Read the user's question and pick EXACTLY ONE route.

        ROUTES:

        rag
        - General questions not covered by the routes below
        - e.g. "What is the customer's email?", "Tell me about this project"

        summary
        - Full project overview / status snapshot requests
        - e.g. "Give me a summary of project X", "What's the project status?"

        cost
        - Budget, price, BOQ value, estimated value, cost overrun, financial range queries
        - e.g. "Cost > 10 lakh", "Show cost overruns", "Budget vs actual"

        zone_analysis
        - Multi-project comparison across zones/regions/cities (qualitative)
        - e.g. "Compare north vs east zone performance", "Analyze all zones"

        flow_analysis
        - ANY question about project PHASE, STAGE, SUBSTAGE, MILESTONE, TIMELINE, SCHEDULE, FLOW or PROGRESS of a SPECIFIC project
        - e.g. "What phase is project X in?", "What substage is this project on?",
              "When will it complete?", "Show me the flow", "What stage are we at?"

        aggregation
        - Counting, breakdowns, "which X has the most/least", trend/compare across time or groups,
          ageing/overdue/stuck/idle counts — anything needing an exact count or group-by across MANY projects
        - e.g. "How many active projects in each zone?", "Which zone has the most projects stuck in design stage?",
              "Zone-wise count of projects pending milestone payment", "City-wise average project timeline"

        IMPORTANT:
        - Questions mentioning "stage", "substage", "phase", "milestone", "timeline", or "progress" for a SPECIFIC project ALWAYS go to flow_analysis.
        - Questions asking "how many", "count of", "which zone/city has the most", or any zone-wise/city-wise breakdown ALWAYS go to aggregation, even if they also mention stage/cost/zone words.

        Return ONLY the route word in lowercase. No punctuation, no explanation, no extra text.`),
        new HumanMessage(state.question),
      ]);

      const VALID_ROUTES = ["rag", "summary", "cost", "zone_analysis", "flow_analysis", "aggregation"];

      const rawRoute = response.content;
      let route = "";

      if (Array.isArray(rawRoute)) {
        route = rawRoute
          .map((item) => {
            if (typeof item === "string") return item;
            if (item && typeof item === "object" && "text" in item) return item.text;
            return "";
          })
          .join(" ");
      } else {
        route = rawRoute?.toString?.() || "";
      }

      route = route.trim().toLowerCase().replace(/[^a-z_]/g, "");

      if (!VALID_ROUTES.includes(route)) {
        console.warn(`⚠️ Supervisor returned invalid route: "${route}", defaulting to rag`);
        route = "rag";
      }

      console.log(`🧭 Supervisor routed to: ${route}`);
      console.log("supervisorAgent response:", response.content);

      return {
        route,
      };
    };

    const ragAgent = async (state: GraphState) => {
      try {
        console.log("🤖 RAG Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("rag")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result =
          typeof response.content === "string"
            ? response.content
            : JSON.stringify(response.content);

        console.log("✅ RAG response generated");

        return {
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ RAG Agent error:", error);
        return {
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const summaryAgent = async (state: GraphState) => {
      try {
        console.log("📋 Summary Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("summary")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          summary: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Summary Agent error:", error);
        return {
          summary: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const costAgent = async (state: GraphState) => {
      try {
        console.log("💰 Cost Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("cost_analysis")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          costAnalysis: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Cost Agent error:", error);
        return {
          costAnalysis: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const flowAnalysisAgent = async (state: GraphState) => {
      try {
        console.log("🗺️ Flow Analysis Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("flow_analysis")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          flowAnalysis: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Flow Analysis Agent error:", error);
        return {
          flowAnalysis: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const zoneAnalysisAgent = async (state: GraphState) => {
      try {
        console.log("🗺️ Zone Analysis Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("zone_analysis")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          zoneAnalysis: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Zone Analysis Agent error:", error);
        return {
          zoneAnalysis: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    // ============ NEW: AGGREGATION AGENT ============
    // Context was already computed exactly (via scroll) in retrieveContext;
    // this node just asks the LLM to phrase the exact numbers in words.

    const aggregationAgent = async (state: GraphState) => {
      try {
        console.log("📊 Aggregation Agent processing...");

        const compoundInstruction = (state.subQuestions?.length || 0) > 1
          ? "\nThe user asked multiple questions in one prompt. Answer each sub-question separately and clearly. If any part cannot be answered from the context, say that explicitly rather than guessing."
          : "";

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("aggregation")}${compoundInstruction}
 
          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          aggregationResult: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Aggregation Agent error:", error);
        return {
          aggregationResult: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const saveCache = async (state: GraphState) => {
      if (!state.result) {
        return {};
      }

      try {
        await this.qdrant.upsert("chat_cache", [
          {
            id: crypto.randomUUID(),
            vector: state.embedding,
            payload: {
              question: state.question,
              normalizedQuestion: state.normalizedQuestion,
              answer: state.result,
              customerId: state.customerId,
              createdAt: new Date().getTime(),
            },
          },
        ]);
        console.log("✅ Response cached");
      } catch (error) {
        console.error("❌ Cache save error:", error);
      }

      return {};
    };

    const routeDecision = (state: GraphState) => {
      switch (state.route) {
        case "summary":
          return "summary";
        case "cost":
          return "cost";
        case "zone_analysis":
          return "zone_analysis";
        case "flow_analysis":
          return "flow_analysis";
        case "aggregation":
          return "aggregation";
        default:
          return "rag";
      }
    };

    // ============ 13. BUILD WORKFLOW ============

    const workflow = new StateGraph(GraphAnnotation)
      .addNode("normalize", normalizeQuery)
      .addNode("embeddings", generateEmbedding)
      .addNode("retrieve", retrieveContext)
      .addNode("supervisor", supervisorAgent)
      .addNode("rag", ragAgent)
      .addNode("summaryV1", summaryAgent)
      .addNode("cost", costAgent)
      .addNode("flow_analysis", flowAnalysisAgent)
      .addNode("zone_analysis", zoneAnalysisAgent)
      .addNode("aggregation", aggregationAgent)
      .addNode("saveCache", saveCache)
      .addEdge(START, "normalize")
      .addEdge("normalize", "embeddings")
      .addEdge("embeddings", "retrieve")
      .addEdge("retrieve", "supervisor")
      .addConditionalEdges("supervisor", routeDecision, {
        rag: "rag",
        summary: "summaryV1",
        cost: "cost",
        flow_analysis: "flow_analysis",
        zone_analysis: "zone_analysis",
        aggregation: "aggregation",
      })
      .addEdge("rag", "saveCache")
      .addEdge("summaryV1", "saveCache")
      .addEdge("cost", "saveCache")
      .addEdge("flow_analysis", "saveCache")
      .addEdge("zone_analysis", "saveCache")
      .addEdge("aggregation", "saveCache")
      .addEdge("saveCache", END);

    const memory = new MemorySaver();
    return workflow.compile({ checkpointer: memory });
  }
}
