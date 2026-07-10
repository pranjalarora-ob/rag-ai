import { BadRequestException, Injectable, Logger } from '@nestjs/common';
import { PgService } from './pg.service';
import { WatermarkStore } from './watermark.store';
import { OpenaiService } from '../openai.service';
import { QdrantService } from '../qdrant.service';
import { SOURCES } from './sources';
import { IngestDocument, SourceDefinition } from './source.types';

const ZERO_UUID = '00000000-0000-0000-0000-000000000000';

/**
 * The ingestion engine — source-agnostic. It pages through a source with keyset
 * pagination, builds documents via the source's toDocument(), batch-embeds, and
 * upserts to Qdrant. Stable ids (projectId/boqId) make every upsert idempotent.
 */
@Injectable()
export class IngestionService {
  private readonly logger = new Logger(IngestionService.name);

  constructor(
    private readonly pg: PgService,
    private readonly openai: OpenaiService,
    private readonly qdrant: QdrantService,
    private readonly watermarks: WatermarkStore,
  ) {}

  private source(name: string): SourceDefinition {
    const src = SOURCES[name];
    if (!src) {
      throw new BadRequestException(
        `Unknown source '${name}'. Available: ${Object.keys(SOURCES).join(', ')}`,
      );
    }
    return src;
  }

  private sinceMonths(months: number): string {
    const d = new Date();
    d.setMonth(d.getMonth() - months);
    return d.toISOString();
  }

  /** Backfill the last N months (default 6) for a source. */
  async backfill(name: string, months = 6) {
    const src = this.source(name);
    await this.ensureIndexes(src.collection);
    const since = this.sinceMonths(months);
    const res = await this.run(src, since);
    if (res.maxUpdatedAt) this.watermarks.set(src.name, res.maxUpdatedAt);
    return { mode: 'backfill', since, ...res };
  }

  /** Incremental: only rows changed since the last successful sync. */
  async sync(name: string) {
    const src = this.source(name);
    await this.ensureIndexes(src.collection);
    const since = this.watermarks.get(src.name) || this.sinceMonths(6);
    const res = await this.run(src, since);
    if (res.maxUpdatedAt) this.watermarks.set(src.name, res.maxUpdatedAt);
    return { mode: 'sync', since, ...res };
  }

  /** Real-time: re-ingest a single entity by id (call from a webhook/queue on create/update). */
  async ingestEntity(name: string, id: string) {
    const src = this.source(name);
    await this.ensureIndexes(src.collection);
    const rows = await this.pg.query(src.db, src.byIdSql, [id]);
    if (!rows.length) return { mode: 'entity', source: src.name, id, count: 0 };
    await this.embedAndUpsert(src, rows.map((r) => src.toDocument(r)));
    return { mode: 'entity', source: src.name, id, count: rows.length };
  }

  // ---- core paging loop ----
  private async run(src: SourceDefinition, sinceISO: string, batchSize = 500) {
    let lastId = ZERO_UUID;
    let total = 0;
    let maxUpdatedAt: string | null = null;

    for (;;) {
      const rows = await this.pg.query(src.db, src.pageSql, [sinceISO, lastId, batchSize]);
      if (!rows.length) break;

      await this.embedAndUpsert(src, rows.map((r) => src.toDocument(r)));

      for (const r of rows) {
        const u = new Date(r.updated_at).toISOString();
        if (!maxUpdatedAt || u > maxUpdatedAt) maxUpdatedAt = u;
      }
      lastId = rows[rows.length - 1].id;
      total += rows.length;
      this.logger.log(`[${src.name}] ingested ${total} rows...`);

      if (rows.length < batchSize) break;
    }
    return { source: src.name, count: total, maxUpdatedAt };
  }

  // ---- chunk -> batch embed -> upsert ----
  private async embedAndUpsert(src: SourceDefinition, docs: IngestDocument[]) {
    if (!docs.length) return;

    const items: { doc: IngestDocument; chunkIndex: number; total: number; text: string }[] = [];
    for (const doc of docs) {
      const chunks = this.qdrant.chunkTextv2(doc.text);
      chunks.forEach((text, chunkIndex) =>
        items.push({ doc, chunkIndex, total: chunks.length, text }),
      );
    }

    const vectors = await this.openai.generateEmbeddings(items.map((it) => it.text));

    const points = items.map((it, k) => ({
      id: this.qdrant.generateChunkId(it.doc.id, it.chunkIndex),
      vector: vectors[k],
      payload: {
        ...it.doc.metadata,
        chunk_index: it.chunkIndex,
        total_chunks: it.total,
        text: it.text,
        original_id: it.doc.id,
      },
    }));

    await this.qdrant.upsert(src.collection, points);
  }

  // Payload indexes needed for fast filtering; ignore "already exists" errors.
  private async ensureIndexes(collection: string) {
    await this.qdrant.createCollection(collection);
    const indexes: { field: string; schema: any }[] = [
      { field: 'docType', schema: 'keyword' },
      { field: 'type', schema: 'keyword' },
      { field: 'active', schema: 'bool' },
      { field: 'projectId', schema: 'keyword' },
      { field: 'accountId', schema: 'keyword' },
      { field: 'customerId', schema: 'keyword' },
      { field: 'projectCode', schema: 'keyword' },
      { field: 'projectCode', schema: 'integer' },
      { field: 'city', schema: 'keyword' },
      { field: 'stage', schema: 'keyword' },
      { field: 'owner', schema: 'keyword' },
      { field: 'areaSft', schema: 'float' },
      { field: 'estimatedValue', schema: 'float' },
      { field: 'team[].userId', schema: 'keyword' },
    ];
    for (const ix of indexes) {
      try {
        await this.qdrant.addIndex(collection, ix as any);
      } catch {
        /* index already exists */
      }
    }
  }
}
