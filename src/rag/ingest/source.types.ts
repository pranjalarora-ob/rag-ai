import { DbName } from './pg.service';

export interface IngestDocument {
  id: string;
  text: string;
  metadata: Record<string, any>;
}

/**
 * A source = one "root" entity (project, boq) that becomes one document per row.
 *
 * To add a new child table later you touch ONLY the source file:
 *   1. add a `json_agg` / sub-select for the child inside its `select()` body
 *   2. add one line in `toDocument()` to fold that child into the text/metadata
 * The engine (ingestion.service) never changes.
 *
 * Timestamps/ids are aliased inside the SQL to the canonical names
 * `id`, `created_at`, `updated_at` so the engine can read them uniformly,
 * even though lead-service uses snake_case and boq-service uses camelCase columns.
 */
export interface SourceDefinition {
  name: string;            // 'project' | 'boq' — used in the API path
  db: DbName;              // which database pool
  collection: string;      // target Qdrant collection

  /** Keyset page: params are [sinceISO, lastId, limit]; must ORDER BY id ASC. */
  pageSql: string;
  /** Single entity (real-time/webhook): param is [id]. */
  byIdSql: string;

  toDocument(row: any): IngestDocument;
}
