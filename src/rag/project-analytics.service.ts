import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { COLLECTION } from './constants';

type Metric = 'estimatedValue' | 'boqValue';

// Each metric maps to a document type + the numeric payload field to aggregate.
// 'boqValue' = the BOQ amount (cost) on boq docs, which now carry the project name.
const METRIC_CONFIG: Record<Metric, { docType: 'project' | 'boq'; field: string }> = {
  estimatedValue: { docType: 'project', field: 'estimatedValue' },
  boqValue: { docType: 'boq', field: 'cost' },
};

export interface AnalyticsQuery {
  customerId: string;
  metric?: Metric;
  topN?: number;
  teamMember?: string;
  role?: string;
  groupBy?: 'stage' | 'city' | 'owner';
  city?: string;
  zone?: string;
  stage?: string;
  owner?: string;
  minValue?: number;
  maxValue?: number;
}

/**
 * Exact aggregates over the vector store — NO text-to-SQL. Reads all of a customer's
 * project points from Qdrant and computes totals/top-N/filtered sums in code.
 */
@Injectable()
export class ProjectAnalyticsService {
  constructor(private readonly qdrantService: QdrantService) {}

  private num(x: any): number {
    const n = Number(x);
    return Number.isFinite(n) ? n : 0;
  }

  private formatNumber(val: number, raw: any): string {
    if (typeof raw === 'string' && /^\d+$/.test(raw)) {
      try {
        const bigintVal = BigInt(raw);
        if (raw.length > 21) {
          const str = bigintVal.toString();
          return `${str[0]}.${str.slice(1, 3)}e+${str.length - 1}`;
        }
        return bigintVal.toLocaleString('en-IN');
      } catch {
        // Fall back
      }
    }
    if (val > Number.MAX_SAFE_INTEGER || val < -Number.MAX_SAFE_INTEGER) {
      return val.toExponential(2);
    }
    return val.toLocaleString('en-IN');
  }

  // Load one payload per document (deduped across chunks) for a given docType.
  private async loadDocs(customerId: string, docType: 'project' | 'boq') {
    const points = await this.qdrantService.scrollAll(COLLECTION, {
      must: [
        { key: 'customerId', match: { value: customerId } },
        { key: 'docType', match: { value: docType } },
      ],
    });
    const byDoc = new Map<string, any>();
    for (const pt of points) {
      const p = pt.payload || {};
      const key = p.original_id || p.boqId || p.projectId;
      if (key && !byDoc.has(key)) byDoc.set(key, p);
    }
    return [...byDoc.values()];
  }

  private matchesTeam(project: any, member: string, role?: string): boolean {
    const team: any[] = [...(project.team || []), ...(project.pocTeam || [])];
    const m = member.trim().toLowerCase();
    return team.some((t) => {
      const nameHit = (t.name || '').trim().toLowerCase().includes(m);
      const roleHit = !role || (t.role || '').toLowerCase().includes(role.trim().toLowerCase());
      return nameHit && roleHit;
    });
  }

  async analyze(query: AnalyticsQuery) {
    const metric: Metric = query.metric || 'estimatedValue';
    const { docType, field } = METRIC_CONFIG[metric];
    const topN = query.topN ?? 5;

    let docs = await this.loadDocs(query.customerId, docType);

    // Filter by city
    if (query.city) {
      const c = query.city.trim().toLowerCase();
      docs = docs.filter((p) => (p.city || '').trim().toLowerCase() === c);
    }

    // Filter by zone
    if (query.zone) {
      const z = query.zone.trim().toLowerCase();
      docs = docs.filter((p) => (p.zone || '').trim().toLowerCase().includes(z));
    }

    // Filter by stage
    if (query.stage) {
      const s = query.stage.trim().toLowerCase();
      docs = docs.filter((p) => (p.stage || '').trim().toLowerCase() === s);
    }

    // Filter by owner
    if (query.owner) {
      const o = query.owner.trim().toLowerCase();
      docs = docs.filter((p) => (p.owner || '').trim().toLowerCase() === o);
    }

    // Filter by value range
    if (query.minValue !== undefined) {
      docs = docs.filter((p) => this.num(p[field]) >= query.minValue);
    }
    if (query.maxValue !== undefined) {
      docs = docs.filter((p) => this.num(p[field]) <= query.maxValue);
    }

    // Team filtering only applies to project docs (BOQ docs have no team).
    if (query.teamMember && docType === 'project') {
      docs = docs.filter((p) => this.matchesTeam(p, query.teamMember!, query.role));
    }

    const rows = docs.map((p) => {
      const val = this.num(p[field]);
      return {
        projectCode: p.projectCode,
        companyName: (p.companyName || '').trim(),
        projectName: p.projectName || null,
        projectId: p.projectId,
        boqCode: docType === 'boq' ? p.boqCode : undefined,
        status: docType === 'boq' ? p.status : undefined,
        city: p.city,
        stage: p.stage,
        owner: p.owner,
        value: val,
        formattedValue: this.formatNumber(val, p[field]),
      };
    });

    const totalVal = rows.reduce((sum, r) => sum + r.value, 0);
    const top = [...rows].sort((a, b) => b.value - a.value).slice(0, topN);

    let breakdown: Record<string, { count: number; total: number }> | undefined;
    if (query.groupBy) {
      breakdown = {};
      for (const r of rows) {
        const key = String((r as any)[query.groupBy!] ?? 'Unknown');
        breakdown[key] = breakdown[key] || { count: 0, total: 0 };
        breakdown[key].count += 1;
        breakdown[key].total += r.value;
      }
    }

    return {
      metric,
      filter: {
        teamMember: query.teamMember || null,
        role: query.role || null,
        city: query.city || null,
        stage: query.stage || null,
        owner: query.owner || null,
      },
      count: rows.length,
      total: totalVal,
      formattedTotal: this.formatNumber(totalVal, null),
      average: rows.length ? totalVal / rows.length : 0,
      formattedAverage: this.formatNumber(rows.length ? totalVal / rows.length : 0, null),
      top,
      breakdown,
    };
  }
}
