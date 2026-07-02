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
}

type Route = 'analytics' | 'lookup' | 'search';

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
  ) {
    this.graph = this.buildGraph();
  }

  // ============ Public API ============

  /** Run the graph to completion and return the final answer + trace. */
  async run(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
  }): Promise<AgentGraphResult> {
    const final = await this.graph.invoke(
      {
        question: params.question,
        customerId: params.customerId,
        history: params.history ?? [],
      },
      { configurable: { thread_id: crypto.randomUUID() } },
    );

    return {
      answer: final.answer ?? '',
      route: final.route ?? 'analytics',
      trace: final.trace ?? [],
      cached: !!final.cached,
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
    history?: Array<{ role: string; content: string }>;
    res: Response;
    onAnswer?: (answer: string) => void;
  }): Promise<void> {
    const { res, onAnswer } = params;
    const { answer } = await this.run(params);
    onAnswer?.(answer);

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

  private cacheSaveNode = async (state: GraphState) => {
    if (!state.cached && !state.blocked && state.answer && state.cacheEmbedding?.length) {
      this.semanticCache
        .save(state.question, state.cacheEmbedding, state.answer, state.customerId)
        .catch(() => {});
    }
    return {};
  };

  // ============ Routing helpers ============

  private async classify(question: string): Promise<Route> {
    // Deterministic disambiguation first — a specific field filter can't be an aggregate.
    const q = question.toLowerCase();
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
      })
      .addEdge('analytics', 'cacheSave')
      .addEdge('lookup', 'cacheSave')
      .addEdge('search', 'cacheSave')
      .addEdge('cacheSave', END);

    return workflow.compile({ checkpointer: new MemorySaver() });
  }
}
