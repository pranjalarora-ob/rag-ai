import { Injectable, OnModuleInit } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';

const CACHE_COLLECTION = 'semantic_cache';
const SCORE_THRESHOLD = 0.92;
const CACHE_TTL_MS = 3 * 60 * 60 * 1000; // 1 hour in milliseconds

@Injectable()
export class SemanticCacheService implements OnModuleInit {
  private memoryCache = new Map<string, { answer: string; expiresAt: number }>();

  constructor(
    private readonly qdrantService: QdrantService,
    private readonly openaiService: OpenaiService,
  ) { }

  async onModuleInit() {
    try {
      await this.qdrantService.createCollection(CACHE_COLLECTION);
      await this.qdrantService.addIndex(CACHE_COLLECTION, { field: 'cachedAt', schema: 'integer' }).catch(() => { });
      await this.qdrantService.addIndex(CACHE_COLLECTION, { field: 'customerId', schema: 'keyword' }).catch(() => { });
      console.log('[SemanticCache] Qdrant collection and payload indexes ensured.');
    } catch (err: any) {
      console.error('[SemanticCache] Failed to initialize Qdrant cache index:', err?.message || err);
    }
  }

  async check(question: string, customerId: string): Promise<{ answer: string; embedding: number[] } | null> {
    const trimmedQuestion = question.trim();
    const cacheKey = `${customerId}:${trimmedQuestion.toLowerCase()}`;

    // 1. Fast-path: Check local in-memory cache first to bypass OpenAI and Qdrant entirely for identical questions
    const cached = this.memoryCache.get(cacheKey);
    if (cached && cached.expiresAt > Date.now()) {
      console.log(`[SemanticCache] MEMORY HIT q="${trimmedQuestion.slice(0, 60)}"`);
      return { answer: cached.answer, embedding: [] };
    }

    // Clean up expired memory cache entries to prevent memory leaks
    if (cached && cached.expiresAt <= Date.now()) {
      this.memoryCache.delete(cacheKey);
    }

    // 2. Slow-path: Generate embedding and query Qdrant for semantic match
    const embedding = await this.openaiService.generateEmbedding(trimmedQuestion);
    const threeHoursAgo = Date.now() - CACHE_TTL_MS;

    console.log(`[SemanticCache] Checking cache for q="${trimmedQuestion.slice(0, 60)}" customerId="${customerId}"`);
    let hits: any[] = [];
    try {
      hits = await this.qdrantService.search(CACHE_COLLECTION, {
        vector: embedding,
        limit: 3,
        score_threshold: SCORE_THRESHOLD,
        with_payload: true, // Explicitly request payload fields
        filter: {
          must: [
            { key: 'customerId', match: { value: customerId } },
            { key: 'cachedAt', range: { gte: threeHoursAgo } },
          ],
        },
      });
      console.log(`[SemanticCache] Qdrant search returned ${hits.length} hits.`);
    } catch (err: any) {
      console.error('[SemanticCache] Qdrant search error:', err?.response?.data || err?.message || err);
      return { answer: null, embedding };
    }

    if (hits.length) {
      for (const [idx, hit] of hits.entries()) {
        console.log(`[SemanticCache] Hit #${idx + 1}: ${JSON.stringify(hit)}`);
      }

      if (hits[0]?.payload?.answer) {
        const cachedQuestion = String(hits[0].payload.question || '');
        if (this.sameIdTokens(trimmedQuestion, cachedQuestion)) {
          console.log(`[SemanticCache] HIT score=${hits[0].score?.toFixed(3)} q="${trimmedQuestion.slice(0, 60)}"`);

          // Populate to local memory cache for faster subsequent exact lookups (expires in 1 hour)
          this.memoryCache.set(cacheKey, {
            answer: hits[0].payload.answer,
            expiresAt: Date.now() + CACHE_TTL_MS,
          });

          return { answer: hits[0].payload.answer, embedding };
        }
        console.log(
          `[SemanticCache] SKIP (id mismatch) score=${hits[0].score?.toFixed(3)} ` +
          `q="${trimmedQuestion.slice(0, 60)}" cached="${cachedQuestion.slice(0, 60)}"`,
        );
      }
    }

    console.log(`[SemanticCache] MISS q="${trimmedQuestion.slice(0, 60)}"`);
    return { answer: null, embedding };
  }

  // Significant numeric tokens (3+ digits) — project codes, ids, filter values.
  // Small numbers like "top 5" / "top 10" are ignored so they don't over-segment.
  private idTokens(q: string): Set<string> {
    return new Set(q.match(/\d{3,}/g) || []);
  }

  // True when two questions carry the same set of significant numeric tokens.
  private sameIdTokens(a: string, b: string): boolean {
    const ta = this.idTokens(a);
    const tb = this.idTokens(b);
    if (ta.size !== tb.size) return false;
    for (const t of ta) if (!tb.has(t)) return false;
    return true;
  }

  // "I don't have that data" style non-answers must never be cached — otherwise a
  // transient miss (bad route, stale code, empty tool result) gets frozen and served
  // back forever, even after the underlying capability is fixed.
  private isLowValueAnswer(answer: string): boolean {
    const a = answer.toLowerCase();
    return /\b(i\s+(don'?t|do not)\s+have|i\s+(couldn'?t|could not|cannot|can'?t)\s+(find|answer|provide)|no\s+(relevant\s+)?(data|information|records?|results?|matches?)|not\s+available|don'?t\s+have\s+(specific|that|any|access)|unable\s+to)\b/.test(a);
  }

  async save(question: string, embedding: number[], answer: string, customerId: string): Promise<void> {
    if (!answer?.trim()) return;
    if (this.isLowValueAnswer(answer)) {
      console.log(`[SemanticCache] SKIP save (non-answer) q="${question.slice(0, 60)}"`);
      return;
    }

    const trimmedQuestion = question.trim();
    const now = Date.now();

    // 1. Save to local in-memory cache (expires in 1 hour)
    const cacheKey = `${customerId}:${trimmedQuestion.toLowerCase()}`;
    this.memoryCache.set(cacheKey, {
      answer,
      expiresAt: now + CACHE_TTL_MS,
    });
    console.log(`[SemanticCache] MEMORY SAVED q="${trimmedQuestion.slice(0, 60)}"`);

    // 2. Save to Qdrant semantic-vector cache
    if (!embedding?.length) return;
    try {
      await this.qdrantService.upsert(CACHE_COLLECTION, [
        {
          id: crypto.randomUUID(),
          vector: embedding,
          payload: { question: trimmedQuestion, answer, customerId, cachedAt: now },
        },
      ]);
      console.log(`[SemanticCache] SAVED q="${trimmedQuestion.slice(0, 60)}"`);
    } catch (err) {
      console.error('[SemanticCache] save error:', (err as any)?.message);
    }

    // 3. Background cleanup of expired cache entries (older than 1 hour)
    const oneHourAgo = now - CACHE_TTL_MS;
    this.qdrantService.deletePoints(CACHE_COLLECTION, {
      must: [
        { key: 'cachedAt', range: { lt: oneHourAgo } },
      ],
    }).catch((err) => {
      console.error('[SemanticCache] background cleanup error:', err?.message);
    });
  }
}
