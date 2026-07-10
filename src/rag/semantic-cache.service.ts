import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';

const CACHE_COLLECTION = 'semantic_cache';
const SCORE_THRESHOLD = 0.95;

@Injectable()
export class SemanticCacheService {
  constructor(
    private readonly qdrantService: QdrantService,
    private readonly openaiService: OpenaiService,
  ) { }

  async check(question: string, customerId: string): Promise<{ answer: string; embedding: number[] } | null> {
    const embedding = await this.openaiService.generateEmbedding(question.trim());

    let hits: any[] = [];
    try {
      hits = await this.qdrantService.search(CACHE_COLLECTION, {
        vector: embedding,
        limit: 1,
        score_threshold: SCORE_THRESHOLD,
        filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
      });
    } catch {
      // Collection may not exist yet on first run — treat as cache miss
      return { answer: null, embedding };
    }

    if (hits.length && hits[0]?.payload?.answer) {
      // Guard against semantic collisions on identifier-bearing queries. Questions
      // that differ ONLY by a project code / id / filter value embed almost
      // identically (cosine > 0.92), so a vector hit alone would happily return
      // project 2640's answer for a question about project 2657. Require the
      // significant numeric tokens (codes, ids, filter numbers — 3+ digits) to
      // match exactly; otherwise treat it as a miss. Pure paraphrases with no
      // numbers still benefit from the cache.
      const cachedQuestion = String(hits[0].payload.question || '');
      if (this.sameIdTokens(question, cachedQuestion)) {
        console.log(`[SemanticCache] HIT score=${hits[0].score?.toFixed(3)} q="${question.slice(0, 60)}"`);
        return { answer: hits[0].payload.answer, embedding };
      }
      console.log(
        `[SemanticCache] SKIP (id mismatch) score=${hits[0].score?.toFixed(3)} ` +
        `q="${question.slice(0, 60)}" cached="${cachedQuestion.slice(0, 60)}"`,
      );
    }

    console.log(`[SemanticCache] MISS q="${question.slice(0, 60)}"`);
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

  async save(question: string, embedding: number[], answer: string, customerId: string): Promise<void> {
    if (!answer?.trim() || !embedding?.length) return;
    try {
      await this.qdrantService.createCollection(CACHE_COLLECTION);
      await this.qdrantService.upsert(CACHE_COLLECTION, [
        {
          id: crypto.randomUUID(),
          vector: embedding,
          payload: { question, answer, customerId, cachedAt: new Date().toISOString() },
        },
      ]);
      console.log(`[SemanticCache] SAVED q="${question.slice(0, 60)}"`);
    } catch (err) {
      console.error('[SemanticCache] save error:', (err as any)?.message);
    }
  }
}
