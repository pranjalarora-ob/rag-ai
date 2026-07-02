import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { COLLECTION } from './constants';

export type RangeOp = 'gt' | 'gte' | 'lt' | 'lte' | 'eq';
export interface RangeCond {
  op: RangeOp;
  value: number;
}

// The structured query the agent fills in from a natural-language question.
export interface ProjectQuerySpec {
  type?: 'lead' | 'project';
  city?: string;
  owner?: string;
  zone?: string;
  projectCode?: number;
  area?: RangeCond;
  estimatedValue?: RangeCond;
  sortBy?: 'areaSft' | 'estimatedValue' | 'currentProjectValue' | 'closureValue' | 'projectCode';
  sortDir?: 'asc' | 'desc';
  limit?: number;
  operation?: 'list' | 'count';
}

export interface ProjectQueryResult {
  operation: 'list' | 'count';
  total: number; // total matches before any limit
  shown?: number; // rows returned after limit
  rows?: any[];
  answer: string; // ready-to-return formatted text (exact, no LLM)
}

/**
 * Deterministic project/lead query engine. Takes a structured spec (filters + sort +
 * limit + operation) and runs it against Qdrant with exact results — no LLM in the loop,
 * so counts/lists are never truncated or reordered. The agent produces the spec; this
 * executes it.
 */
@Injectable()
export class ProjectQueryService {
  constructor(private readonly qdrant: QdrantService) {}

  async execute(spec: ProjectQuerySpec, customerId: string): Promise<ProjectQueryResult> {
    const must: any[] = [
      { key: 'customerId', match: { value: customerId } },
      { key: 'docType', match: { value: 'project' } },
    ];
    if (spec.type) must.push({ key: 'type', match: { value: spec.type } });
    // projectCode is stored as a STRING in Qdrant — match as string even though the
    // tool passes a number (integer match would return zero rows).
    if (spec.projectCode) must.push({ key: 'projectCode', match: { value: String(spec.projectCode) } });
    if (spec.area) must.push({ key: 'areaSft', range: this.toRange(spec.area) });
    if (spec.estimatedValue) must.push({ key: 'estimatedValue', range: this.toRange(spec.estimatedValue) });

    const points = await this.qdrant.scrollAll(COLLECTION, { must });

    // city/zone matched case-insensitively, owner typo-tolerantly (mixed-casing data).
    const cityLc = spec.city?.toLowerCase();
    const zoneLc = spec.zone?.toLowerCase();
    const seen = new Set<string>();
    let items: any[] = [];
    for (const p of points) {
      const pl: any = p.payload || {};
      const id = pl.original_id || pl.projectId;
      if (!id || seen.has(id)) continue;
      if (cityLc && String(pl.city || '').toLowerCase() !== cityLc) continue;
      if (spec.owner && !this.fuzzyTextMatch(String(pl.owner || ''), spec.owner)) continue;
      if (zoneLc && !String(pl.zone || '').toLowerCase().includes(zoneLc)) continue;
      seen.add(id);
      items.push(pl);
    }

    const total = items.length;
    const label = spec.type ? `${spec.type}(s)` : 'project(s)';

    if (spec.operation === 'count') {
      return { operation: 'count', total, answer: `${total} ${label} match the filter.` };
    }

    // Exact code lookup -> return the FULL project summary text (for "tell me about ...").
    if (spec.projectCode && items.length) {
      const answer = items
        .map((p) => p.text)
        .filter(Boolean)
        .join('\n\n---\n\n');
      return { operation: 'list', total, shown: items.length, rows: items, answer };
    }

    if (spec.sortBy) {
      const dir = spec.sortDir === 'asc' ? 1 : -1;
      const key = spec.sortBy;
      items.sort((a, b) => ((Number(a[key]) || 0) - (Number(b[key]) || 0)) * dir);
    }
    if (spec.limit) items = items.slice(0, spec.limit);

    let answer: string;
    if (!items.length) {
      answer = 'No projects match that filter.';
    } else {
      const sortNote = spec.sortBy
        ? ` (sorted by ${spec.sortBy} ${spec.sortDir === 'asc' ? 'ascending' : 'descending'})`
        : '';
      const header = spec.limit
        ? `Top ${items.length} ${label} of ${total} matching${sortNote}:`
        : `${total} matching ${label}${sortNote}:`;
      const head = '| # | Code | Name | City | Zone | Owner | Area (sqft) | Estimated Value |\n|---|---|---|---|---|---|---|---|';
      const body = items
        .map(
          (pl, i) =>
            `| ${i + 1} | ${pl.projectCode} | ${pl.projectName || (pl.companyName || '').trim()} | ${pl.city ?? ''} | ${pl.zone ?? ''} | ${pl.owner ?? ''} | ${pl.areaSft ?? ''} | ${pl.estimatedValue ?? ''} |`,
        )
        .join('\n');
      answer = `${header}\n\n${head}\n${body}`;
    }
    return { operation: 'list', total, shown: items.length, rows: items, answer };
  }

  private toRange(c: RangeCond) {
    switch (c.op) {
      case 'gt':
        return { gt: c.value };
      case 'gte':
        return { gte: c.value };
      case 'lt':
        return { lt: c.value };
      case 'lte':
        return { lte: c.value };
      case 'eq':
        return { gte: c.value, lte: c.value };
    }
  }

  // ---- typo-tolerant text match (owner) ----
  private fuzzyTextMatch(value: string, query: string): boolean {
    const v = (value || '').toLowerCase().trim();
    const q = (query || '').toLowerCase().trim();
    if (!q) return true;
    if (!v) return false;
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
          d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
        }
      }
    }
    return d[al][bl];
  }
}
