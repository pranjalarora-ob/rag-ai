import {
  Controller, Post, Body, Param, Get, Res, Patch,
  BadRequestException, InternalServerErrorException,
} from '@nestjs/common';
import { ApiOperation, ApiProduces, ApiResponse, ApiTags } from '@nestjs/swagger';
import { Response } from 'express';
import { ChatDto, IngestVectorDataDto, PlannerDto } from './dto/rag.dto';
import { AddSchemaIndexDto, Point } from './dto/qdrant.dto';
import { ChatCompletionRequestMessage } from './dto/openai.dto';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { ClaudeService } from './claude.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService, AnalyticsQuery } from './project-analytics.service';
import { PlannerService } from './planner.service';
import { RerankService } from './rerank.service';
import { ProjectAgentService } from './project-agent.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

@ApiTags('RAG')
@Controller('rag')
export class RagController {
  constructor(
    private readonly qdrantService: QdrantService,
    private readonly openaiService: OpenaiService,
    private readonly claudeService: ClaudeService,
    private readonly guardrailService: GuardrailService,
    private readonly projectAnalyticsService: ProjectAnalyticsService,
    private readonly plannerService: PlannerService,
    private readonly rerankService: RerankService,
    private readonly projectAgentService: ProjectAgentService,
  ) { }

  // ============ Ingestion ============

  @ApiOperation({ summary: 'Ingest documents into the Qdrant vector DB (chunk + embed)' })
  @Post('openai/qdrant/ingest')
  async ingest(@Body() dto: IngestVectorDataDto) {
    if (!Array.isArray(dto.documents)) {
      throw new BadRequestException("Invalid 'documents' format");
    }

    const points: Point[] = [];
    for (const doc of dto.documents) {
      const chunks = this.qdrantService.chunkTextv2(doc.text);
      for (let i = 0; i < chunks.length; i++) {
        const embedding = await this.openaiService.generateEmbedding(chunks[i]);
        points.push({
          id: this.qdrantService.generateChunkId(doc.id, i),
          vector: embedding,
          payload: {
            ...doc.metadata,
            chunk_index: i,
            customerId: dto.customerId,
            total_chunks: chunks.length,
            text: chunks[i],
            original_id: doc.id,
          },
        });
      }
    }

    await this.qdrantService.upsert(dto.collection, points);
    return { message: 'Ingestion successful', count: points.length };
  }

  @ApiOperation({ summary: 'Clear a Qdrant collection' })
  @Get('qdrant/:collection/clear')
  clearCollection(@Param('collection') collection: string) {
    return this.qdrantService.clearCollection(collection);
  }

  @ApiOperation({ summary: 'Add a payload index (required before filtering on a field, e.g. customerId)' })
  @Patch('qdrant/:collection/add-index')
  addIndex(@Param('collection') collection: string, @Body() body: AddSchemaIndexDto) {
    return this.qdrantService.addIndex(collection, body);
  }

  // ============ Tools ============

  @ApiOperation({ summary: 'Analytics tool — exact totals, top-N, averages, filtered sums (no LLM math)' })
  @Post('projects/analytics')
  analytics(@Body() query: AnalyticsQuery) {
    if (!query?.customerId) throw new BadRequestException('customerId is required');
    return this.projectAnalyticsService.analyze(query);
  }

  // ============ Planner (the thinker) ============

  @ApiOperation({ summary: 'Planner agent — picks tools (RAG / analytics), loops, drafts answer. Returns answer + trace.' })
  @Post('planner')
  planner(@Body() body: PlannerDto) {
    return this.plannerService.run({ question: body.question, customerId: body.customerId });
  }

  @ApiOperation({ summary: 'Agent — LLM turns the question into a structured query, run deterministically (exact filters/sort/count). Returns answer + trace.' })
  @Post('agent')
  agent(@Body() body: PlannerDto) {
    if (!body?.customerId) throw new BadRequestException('customerId is required');
    return this.projectAgentService.run({ question: body.question, customerId: body.customerId });
  }

  // ============ Chat (streaming, with routing) ============

  @ApiOperation({ summary: 'Chat — routes aggregate questions to the analytics tool, lookups to RAG. Streams text.' })
  @ApiProduces('text/event-stream')
  @ApiResponse({ status: 200, description: 'Streamed plain-text answer' })
  @Post('openai/chat')
  async chat(@Body() chatDto: ChatDto, @Res() res: Response) {
    const { question, customerId } = chatDto;

    try {
      if (this.guardrailService.isPolicyViolation(question)) {
        throw new BadRequestException('Query violates company policy.');
      }

      // Structured LISTING queries ("all leads in Gurugram", "projects with area > 5000")
      // must use the filter/scroll path — NOT the aggregate analytics tool, which ignores
      // city/type. Detect them up front and force the RAG route. True aggregates
      // ("total", "top 5", "average") still fall through to the analytics classifier.
      const aggregateHint = /\b(total|sum|average|avg|count|how many|highest|lowest|top\s*\d+|rank|most|least|combined)\b/i.test(question);
      // Field filters (city / area / value) can't be done by the analytics tool, so they
      // ALWAYS force the filter/scroll path — even with "top N" (handled as a limit there).
      // A bare type word ("projects"/"leads") forces scroll only when it's not an aggregate.
      const fieldFilter = !!(
        this.parseCity(question) ||
        this.parseOwner(question) ||
        this.parseZone(question) ||
        this.parseNumericRange(question, ['area', 'sqft', 'sft', 'square feet', 'square foot']) ||
        this.parseNumericRange(question, ['estimated value', 'estimatedvalue', 'value', 'budget', 'worth'])
      );
      const route =
        fieldFilter || (this.parseType(question) && !aggregateHint)
          ? { type: 'rag' as const }
          : await this.classifyQuery(question);

      // ---- Analytics route: exact numbers from the tool, phrased by the LLM ----
      if (route.type === 'analytics') {
        const analytics = await this.projectAnalyticsService.analyze({
          customerId,
          metric: route.metric,
          topN: route.topN,
          teamMember: route.teamMember,
          role: route.role,
          groupBy: route.groupBy,
          city: route.city,
          stage: route.stage,
          owner: route.owner,
        });

        const messages: ChatCompletionRequestMessage[] = [
          {
            role: 'system',
            content: `${SYSTEM_PROMPT}\n\nAnswer using these EXACT pre-computed results. Do not recompute or invent numbers. Use the exact string values in the 'formattedValue', 'formattedTotal', or 'formattedAverage' fields for all numeric/financial values when present.\n\nResults (JSON):\n${JSON.stringify(analytics)}`,
          },
          { role: 'user', content: question },
        ];
        await this.streamAnswer(messages, res);
        return;
      }

      // ---- RAG route: retrieve relevant projects, answer from them ----
      // Exact lookups by project code can't be found by vector similarity (one code
      // out of hundreds of near-identical project vectors). If the question names a
      // project code, filter on it directly — projectCode is an indexed payload field.
      const must: any[] = [{ key: 'customerId', match: { value: customerId } }];

      // Structured questions ("area more than 5000", "estimated value over 1 cr",
      // "projects in Gurugram", "list all ...") are filters, not semantics: a vector
      // top-100 can neither evaluate "> N" / "= city" nor return ALL matches.
      // Detect them and use a Qdrant filter + full scroll instead.
      const areaRange = this.parseNumericRange(question, ['area', 'sqft', 'sft', 'square feet', 'square foot']);
      const valueRange = this.parseNumericRange(question, ['estimated value', 'estimatedvalue', 'value', 'budget', 'worth']);
      const city = this.parseCity(question);
      const owner = this.parseOwner(question);
      const zone = this.parseZone(question);

      // A 6+ digit number is a project code ONLY when it's not the operand of an
      // area/value filter (in "area is 200000" the 200000 is the area, not a code).
      const codeMatch = !areaRange && !valueRange ? question.match(/\b(\d{6,})\b/) : null;
      if (codeMatch) must.push({ key: 'projectCode', match: { value: Number(codeMatch[1]) } });

      // "project" / "lead" in the question scopes to that DB type (both are docType:
      // project, split by the `type` column). Skipped for exact code lookups, which
      // resolve to one record and need the full document detail.
      const typeFilter = codeMatch ? null : this.parseType(question);

      if (areaRange || valueRange || city || owner || zone || typeFilter) {
        must.push({ key: 'docType', match: { value: 'project' } });
        if (typeFilter) must.push({ key: 'type', match: { value: typeFilter } });
        if (areaRange) must.push({ key: 'areaSft', range: areaRange });
        if (valueRange) must.push({ key: 'estimatedValue', range: valueRange });

        const points = await this.qdrantService.scrollAll(COLLECTION, { must });
        // city / owner / zone are matched in code (case-insensitive) — the source data has
        // mixed casing, so exact Qdrant keyword matches are unreliable.
        const cityLc = city?.toLowerCase();
        const zoneLc = zone?.toLowerCase();
        const seen = new Set<string>();
        let items: any[] = [];
        for (const p of points) {
          const pl: any = p.payload || {};
          const id = pl.original_id || pl.projectId;
          if (!id || seen.has(id)) continue;
          if (cityLc && String(pl.city || '').toLowerCase() !== cityLc) continue;
          // owner is typo-tolerant (fuzzy); zone stays an exact case-insensitive substring.
          if (owner && !this.fuzzyTextMatch(String(pl.owner || ''), owner)) continue;
          if (zoneLc && !String(pl.zone || '').toLowerCase().includes(zoneLc)) continue;
          seen.add(id);
          items.push(pl);
        }
        const totalMatched = items.length;
        const typeLabel = typeFilter ? `${typeFilter}(s)` : 'project(s)';

        // Count intent ("count of", "how many") -> return just the number, not a table.
        if (/\b(count|how many|number of|no\.?\s*of)\b/i.test(question)) {
          res.setHeader('Content-Type', 'text/plain; charset=utf-8');
          res.send(`${totalMatched} ${typeLabel} match the filter.`);
          return;
        }

        const topN = this.parseTopN(question);
        // "top N" sorts by the dimension being asked about: by area when the question is
        // about area, otherwise by estimated value (explicit value words win).
        const sortKey =
          /\b(value|estimated|cost|budget|worth)\b/i.test(question)
            ? 'estimatedValue'
            : areaRange
              ? 'areaSft'
              : 'estimatedValue';
        items = items.sort((a, b) => (Number(b[sortKey]) || 0) - (Number(a[sortKey]) || 0));
        if (topN) items = items.slice(0, topN);

        // Deterministic list — return the table DIRECTLY, no LLM. Routing an exact list
        // through the model truncates/reorders rows and is slow; the data is already exact.
        const sortLabel = sortKey === 'areaSft' ? 'area' : 'estimated value';
        let answer: string;
        if (!items.length) {
          answer = 'No projects match that filter.';
        } else {
          const header = topN
            ? `Top ${items.length} ${typeLabel} (of ${totalMatched} matching) by ${sortLabel}:`
            : `${totalMatched} matching ${typeLabel}:`;
          const head =
            '| # | Code | Name | City | Area (sqft) | Estimated Value |\n|---|---|---|---|---|---|';
          const body = items
            .map(
              (pl, i) =>
                `| ${i + 1} | ${pl.projectCode} | ${pl.projectName || (pl.companyName || '').trim()} | ${pl.city ?? ''} | ${pl.areaSft ?? ''} | ${pl.estimatedValue ?? ''} |`,
            )
            .join('\n');
          answer = `${header}\n\n${head}\n${body}`;
        }
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.send(answer);
        return;
      }

      const embedding = await this.openaiService.generateEmbedding(question.trim());
      const hits = await this.qdrantService.search(COLLECTION, {
        vector: embedding,
        limit: 100,
        filter: { must },
      });
      const rawChunks = hits.sort((a: any, b: any) => b.score - a.score).map((r: any) => r.payload?.text as string);
      const rerankedChunks = await this.rerankService.rerank(question.trim(), rawChunks, 20);
      const context = await this.openaiService.truncateByTokens(rerankedChunks, 1000);

      const messages: ChatCompletionRequestMessage[] = [
        {
          role: 'system',
          content: `${SYSTEM_PROMPT}\n\nUse ONLY the context below to answer. If it asks for numbers, return exact values from the context; do not invent.\n\nContext:\n${context}`,
        },
        { role: 'user', content: question },
      ];
      await this.streamAnswer(messages, res);
    } catch (err) {
      console.error(err);
      if (!res.headersSent) {
        throw new InternalServerErrorException('Chat service failed');
      }
      if (!res.writableEnded) res.end();
    }
  }

  // ============ Helpers ============

  // Stream the answer via Claude when it's configured, otherwise OpenRouter.
  // If Claude is configured but the upstream call fails before any bytes are
  // sent (bad key, retired model, network), fall back to OpenRouter so the
  // request still succeeds instead of dying with a broken empty response.
  private async streamAnswer(messages: ChatCompletionRequestMessage[], res: Response) {
    if (this.claudeService.isConfigured) {
      const handled = await this.claudeService.streamClaude(messages, res);
      if (handled) return;
    }
    await this.openaiService.streamOpenRouter(messages, res);
  }

  // Extract a numeric condition tied to a field, e.g. "area more than 5000",
  // "estimated value over 100000". `keywords` are the field's aliases in natural
  // language. Returns a Qdrant range object ({ gt }/{ lt }/{ gte }/{ lte }) or null.
  private parseNumericRange(
    question: string,
    keywords: string[],
  ): { gt?: number; lt?: number; gte?: number; lte?: number } | null {
    const q = question.toLowerCase();
    const kw = keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    // Capture an optional Indian/Western unit after the number (lakh/crore/k/million...).
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
    // Equality ("area is 200000", "area of 5000", "area = 1000") -> exact match via gte+lte.
    if ((m = q.match(re('exactly|equal to|equals|is|are|of|=')))) {
      const n = val(m);
      return { gte: n, lte: n };
    }
    return null;
  }

  // "top 5" / "top 10" -> 5 / 10 (a result limit, applied after filtering).
  private parseTopN(question: string): number | null {
    const m = question.match(/\btop\s+(\d+)\b/i);
    return m ? Number(m[1]) : null;
  }

  // "lead" in the question -> only leads; otherwise "project" -> only projects.
  private parseType(question: string): 'lead' | 'project' | null {
    if (/\blead(s)?\b/i.test(question)) return 'lead';
    if (/\bproject(s)?\b/i.test(question)) return 'project';
    return null;
  }

  // Optimal string alignment distance (Levenshtein + adjacent transpositions).
  private osa(a: string, b: string): number {
    const al = a.length;
    const bl = b.length;
    if (!al) return bl;
    if (!bl) return al;
    const d: number[][] = Array.from({ length: al + 1 }, () => new Array(bl + 1).fill(0));
    for (let i = 0; i <= al; i++) d[i][0] = i;
    for (let j = 0; j <= bl; j++) d[0][j] = j;
    for (let i = 1; i <= al; i++) {
      for (let j = 1; j <= bl; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
        if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
          d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1); // transposition
        }
      }
    }
    return d[al][bl];
  }

  // Typo-tolerant text match — substring, or per-token within a small edit distance
  // (so "SBM Team" matches "SMB Team", but unrelated words don't match).
  private fuzzyTextMatch(value: string, query: string): boolean {
    const v = (value || '').toLowerCase().trim();
    const q = (query || '').toLowerCase().trim();
    if (!q) return true;
    if (!v) return false; // empty field value can't match a non-empty query
    if (v.includes(q) || q.includes(v)) return true;
    const vt = v.split(/\s+/);
    const qt = q.split(/\s+/);
    return qt.every((qtok) =>
      vt.some((vtok) => {
        if (vtok.includes(qtok) || qtok.includes(vtok)) return true;
        const thr = Math.max(1, Math.floor(Math.max(vtok.length, qtok.length) * 0.34));
        return this.osa(vtok, qtok) <= thr;
      }),
    );
  }

  // Extract an owner from "owner is SMB Team", "owner SMB Team", "owned by X".
  private parseOwner(question: string): string | null {
    const m = question.match(
      /\b(?:owner|owned by)\s+(?:is\s+|=\s*|:\s*)?([A-Za-z][A-Za-z0-9 .&_-]*?)(?:\s+(?:in|with|where|which|and|having|zone|city|area|stage)\b|[?.,]|$)/i,
    );
    const v = m?.[1]?.trim();
    return v && v.length >= 2 ? v : null;
  }

  // Extract a zone from "zone is North", "zone North".
  private parseZone(question: string): string | null {
    const m = question.match(
      /\bzone\s+(?:is\s+|=\s*|:\s*)?([A-Za-z][A-Za-z0-9 .&_-]*?)(?:\s+(?:in|with|where|which|and|having|owner|city|area|stage)\b|[?.,]|$)/i,
    );
    const v = m?.[1]?.trim();
    return v && v.length >= 2 ? v : null;
  }

  // Extract a city from "projects in Gurugram", "located in Delhi", "city of Mumbai".
  // Heuristic — matched case-insensitively against payload.city at scroll time.
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

  // Decide analytics vs RAG and extract analytics params. Keyword fast-path avoids an
  // extra LLM call for obvious cases; otherwise an LLM classifier extracts name filters.
  private async classifyQuery(question: string): Promise<{
    type: 'analytics' | 'rag';
    metric?: 'estimatedValue' | 'boqValue';
    topN?: number;
    teamMember?: string;
    role?: string;
    groupBy?: 'stage' | 'city' | 'owner';
    city?: string;
    stage?: string;
    owner?: string;
  }> {
    const q = question.toLowerCase();
    const aggregateHint = /\b(total|sum|average|avg|count|how many|highest|lowest|top \d+|top\s|rank|most|least|combined)\b/.test(q);
    const lookupHint = /\b(who|what stage|which stage|describe|details? of|tell me about|status of)\b/.test(q);
    const hasNameFilter = /\b(where|whose|by)\b/.test(q) && /\b(manager|lead|member|owner|head)\b/.test(q);
    const hasFilterHint = /\b(in|for|where|by|stage|city|status|owner|team|member|manager|lead|from)\b/.test(q) &&
      !/\b(by\s+value|by\s+cost|by\s+estimated\s*value|by\s+boq\s*value|by\s+city|by\s+stage|by\s+owner|by\s+team)\b/.test(q);

    if (aggregateHint && !lookupHint && !hasNameFilter && !hasFilterHint) {
      const topMatch = q.match(/top\s+(\d+)/);
      const metric: 'estimatedValue' | 'boqValue' = /\bboq\b/.test(q) ? 'boqValue' : 'estimatedValue';
      let groupBy: 'stage' | 'city' | 'owner' | undefined;
      if (/\bper city|by city|each city\b/.test(q)) groupBy = 'city';
      else if (/\bper stage|by stage|each stage\b/.test(q)) groupBy = 'stage';
      else if (/\bper owner|by owner|each owner|by team\b/.test(q)) groupBy = 'owner';
      return { type: 'analytics', metric, topN: topMatch ? Number(topMatch[1]) : undefined, groupBy };
    }

    const prompt = `You are a router for a project assistant. Classify the user's question and extract parameters.
Return STRICT JSON only — no markdown.

Schema:
{ "type": "analytics" | "rag", "metric": "estimatedValue" | "boqValue", "topN": number, "teamMember": string, "role": string, "groupBy": "stage" | "city" | "owner", "city": string, "stage": string, "owner": string }

Use "analytics" for totals, sums, averages, counts, rankings, or filtered aggregates ACROSS many projects.
Use "rag" for descriptive lookups about a specific project.

Examples:
"total cost of all projects" -> {"type":"analytics","metric":"estimatedValue"}
"top 5 projects by value" -> {"type":"analytics","metric":"estimatedValue","topN":5}
"total estimated value where Ashish is the design manager" -> {"type":"analytics","metric":"estimatedValue","teamMember":"Ashish","role":"Design Manager"}
"details of top 2 project in Gurugram cost wise" -> {"type":"analytics","metric":"estimatedValue","topN":2,"city":"Gurugram"}
"who is the design manager on yatra 87" -> {"type":"rag"}

Question: "${question}"`;

    try {
      // Prefer Claude, but fall back to OpenRouter if it's not configured or the
      // call fails (claudeGenerate returns '' on error) — otherwise every query
      // would silently default to 'rag' whenever Claude is down.
      let raw = this.claudeService.isConfigured
        ? await this.claudeService.claudeGenerate(prompt)
        : '';
      if (!raw) raw = await this.openaiService.openRouterGenerate(prompt);
      const json = raw.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(json);
      if (parsed?.type === 'analytics' || parsed?.type === 'rag') return parsed;
    } catch (err) {
      console.error('classifyQuery failed, defaulting to rag:', err);
    }
    return { type: 'rag' };
  }
}
