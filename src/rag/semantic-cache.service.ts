import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';

const CACHE_COLLECTION = 'semantic_cache';
const SCORE_THRESHOLD = 0.92;

@Injectable()
export class SemanticCacheService {
  constructor(
    private readonly qdrantService: QdrantService,
    private readonly openaiService: OpenaiService,
  ) {}

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
      console.log(`[SemanticCache] HIT score=${hits[0].score?.toFixed(3)} q="${question.slice(0, 60)}"`);
      return { answer: hits[0].payload.answer, embedding };
    }

    console.log(`[SemanticCache] MISS q="${question.slice(0, 60)}"`);
    return { answer: null, embedding };
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
