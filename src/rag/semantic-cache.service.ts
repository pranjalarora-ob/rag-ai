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
      return { answer: hits[0].payload.answer, embedding };
    }

    return { answer: null, embedding };
  }

  async save(question: string, embedding: number[], answer: string, customerId: string): Promise<void> {
    if (!answer?.trim()) return;
    try {
      await this.qdrantService.upsert(CACHE_COLLECTION, [
        {
          id: crypto.randomUUID(),
          vector: embedding,
          payload: { question, answer, customerId, cachedAt: new Date().toISOString() },
        },
      ]);
    } catch (err) {
      // Cache write failure is non-fatal
      console.error('SemanticCache save error:', err?.message);
    }
  }
}
