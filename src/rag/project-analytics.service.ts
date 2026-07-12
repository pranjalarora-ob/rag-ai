import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { COLLECTION } from './constants';
import { toRegion } from './region.util';

type Metric = 'estimatedValue' | 'boqValue' | 'area';

// Each metric maps to a document type + the numeric payload field to aggregate.
// 'boqValue' = the BOQ amount (cost) on boq docs, which now carry the project name.
// 'area' = project area in sqft (clean numeric data, good for charts).
const METRIC_CONFIG: Record<Metric, { docType: 'project' | 'boq'; field: string; isArea?: boolean }> = {
  estimatedValue: { docType: 'project', field: 'estimatedValue' },
  boqValue: { docType: 'boq', field: 'cost' },
  area: { docType: 'project', field: 'areaSft', isArea: true },
};

export interface AnalyticsQuery {
  customerId: string;
  metric?: Metric;
  topN?: number;
  projectCodes?: string[];
  teamMember?: string;
  role?: string;
  groupBy?: 'stage' | 'subStage' | 'city' | 'state' | 'zone' | 'region' | 'owner' | 'channel' | 'projectStatus';
  city?: string;
  state?: string;
  zone?: string;
  region?: string; // normalized region (North/South/East/West/Unassigned)
  stage?: string;
  subStage?: string;
  owner?: string;
  channel?: string;
  projectStatus?: string;
  companyName?: string;
  customerName?: string;
  minValue?: number;
  maxValue?: number;
  equalsValue?: number;
  minArea?: number;
  maxArea?: number;
  equalsArea?: number;
  // ---- lead/project lifecycle + date dimension ----
  type?: 'lead' | 'project'; // filter by record type (lead vs converted project)
  active?: boolean; // only active (or inactive) projects
  ownerMissing?: boolean; // only records with no assigned owner
  dateField?: 'createdAt' | 'updatedAt'; // which date the date filters/buckets use (default createdAt)
  lastNDays?: number; // dateField within the last N days
  lastNMonths?: number; // dateField within the last N months
  createdAfter?: string; // ISO date — dateField on/after
  createdBefore?: string; // ISO date — dateField on/before
  bucketBy?: 'month' | 'quarter'; // add a timeline breakdown grouped by month/quarter of dateField
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
    // The LLM sometimes passes an unsupported metric (e.g. "area"/"cost"); fall back
    // to estimatedValue instead of crashing on an undefined config lookup.
    const metric: Metric = METRIC_CONFIG[query.metric as Metric] ? (query.metric as Metric) : 'estimatedValue';
    const { docType, field } = METRIC_CONFIG[metric];
    const topN = query.topN ?? 50;

    let docs = await this.loadDocs(query.customerId, docType);

    // Exact lookup by project code(s). Normalizes a leading "P" and case so
    // "2022072479", "p2022072479" and "P2022072479" all match the same record.
    if (query.projectCodes?.length) {
      const norm = (c: any) => (c || '').toString().trim().toLowerCase().replace(/^p/, '');
      const wanted = new Set(query.projectCodes.map(norm));
      docs = docs.filter((p) => wanted.has(norm(p.projectCode)));
    }

    // Exact (case-insensitive) equality filters on payload fields.
    const eq = (field: string, val?: string) => {
      if (!val) return;
      const v = val.trim().toLowerCase();
      docs = docs.filter((p) => (p[field] || '').toString().trim().toLowerCase() === v);
    };
    // Substring (case-insensitive) filters — better for free-text-ish fields.
    const includes = (field: string, val?: string) => {
      if (!val) return;
      const v = val.trim().toLowerCase();
      docs = docs.filter((p) => (p[field] || '').toString().trim().toLowerCase().includes(v));
    };

    eq('city', query.city);
    eq('state', query.state);
    includes('zone', query.zone);
    // Region filter: normalize each doc's zone to a canonical region and match.
    if (query.region) {
      const r = query.region.trim().toLowerCase();
      docs = docs.filter((p) => toRegion(p.zone).toLowerCase() === r);
    }
    eq('stage', query.stage);
    eq('subStage', query.subStage);
    eq('owner', query.owner);
    eq('channel', query.channel);
    eq('projectStatus', query.projectStatus);
    includes('companyName', query.companyName);

    // Customer name lives in the nested customerInfo object.
    if (query.customerName) {
      const n = query.customerName.trim().toLowerCase();
      docs = docs.filter((p) => (p.customerInfo?.name || '').toString().trim().toLowerCase().includes(n));
    }

    // Filter by value: exact match takes precedence over range.
    if (query.equalsValue !== undefined) {
      docs = docs.filter((p) => this.num(p[field]) === query.equalsValue);
    } else {
      if (query.minValue !== undefined) {
        docs = docs.filter((p) => this.num(p[field]) >= query.minValue);
      }
      if (query.maxValue !== undefined) {
        docs = docs.filter((p) => this.num(p[field]) <= query.maxValue);
      }
    }

    // Filter by area (areaSft): exact match takes precedence over range.
    if (query.equalsArea !== undefined) {
      docs = docs.filter((p) => this.num(p.areaSft) === query.equalsArea);
    } else {
      if (query.minArea !== undefined) {
        docs = docs.filter((p) => this.num(p.areaSft) >= query.minArea);
      }
      if (query.maxArea !== undefined) {
        docs = docs.filter((p) => this.num(p.areaSft) <= query.maxArea);
      }
    }

    // Team filtering only applies to project docs (BOQ docs have no team).
    if (query.teamMember && docType === 'project') {
      docs = docs.filter((p) => this.matchesTeam(p, query.teamMember!, query.role));
    }

    // ---- lead/project lifecycle filters ----
    if (query.type) {
      const t = query.type.toLowerCase();
      docs = docs.filter((p) => String(p.type || '').toLowerCase() === t);
    }
    if (query.active !== undefined) {
      docs = docs.filter((p) => Boolean(p.active) === query.active);
    }
    if (query.ownerMissing) {
      docs = docs.filter((p) => !String(p.owner || '').trim());
    }

    // ---- date dimension: filter by a date field (default createdAt) ----
    const dateField = query.dateField || 'createdAt';
    const parseDate = (s: any): Date | null => {
      if (!s) return null;
      const d = new Date(s);
      return isNaN(d.getTime()) ? null : d;
    };
    if (query.lastNDays != null) {
      const cut = new Date(Date.now() - query.lastNDays * 86400000);
      docs = docs.filter((p) => { const d = parseDate(p[dateField]); return d !== null && d >= cut; });
    }
    if (query.lastNMonths != null) {
      const cut = new Date();
      cut.setMonth(cut.getMonth() - query.lastNMonths);
      docs = docs.filter((p) => { const d = parseDate(p[dateField]); return d !== null && d >= cut; });
    }
    if (query.createdAfter) {
      const a = parseDate(query.createdAfter);
      if (a) docs = docs.filter((p) => { const d = parseDate(p[dateField]); return d !== null && d >= a; });
    }
    if (query.createdBefore) {
      const b = parseDate(query.createdBefore);
      if (b) docs = docs.filter((p) => { const d = parseDate(p[dateField]); return d !== null && d <= b; });
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
        state: p.state,
        zone: p.zone,
        region: toRegion(p.zone),
        stage: p.stage,
        subStage: p.subStage,
        channel: p.channel,
        projectStatus: p.projectStatus,
        owner: p.owner,
        // Always expose the REAL fields independently of the ranking metric, so the
        // Area and Estimated Value columns never collapse into the same number when
        // ranking by area (field === 'areaSft' would otherwise make value === area).
        area: this.num(p.areaSft),
        estimatedValue: this.num(p.estimatedValue),
        // `value` is the metric being ranked/summed (may equal area when metric==='area').
        value: val,
        formattedValue: this.formatNumber(val, p[field]),
      };
    });

    const totalVal = rows.reduce((sum, r) => sum + r.value, 0);
    const top = [...rows].sort((a, b) => b.value - a.value).slice(0, topN);

    // Timeline: group counts/totals by month or quarter of the date field.
    let timeline: Record<string, { count: number; total: number }> | undefined;
    if (query.bucketBy) {
      timeline = {};
      for (const p of docs) {
        const d = parseDate(p[dateField]);
        if (!d) continue;
        const key =
          query.bucketBy === 'quarter'
            ? `${d.getFullYear()}-Q${Math.floor(d.getMonth() / 3) + 1}`
            : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
        timeline[key] = timeline[key] || { count: 0, total: 0 };
        timeline[key].count += 1;
        timeline[key].total += this.num(p[field]);
      }
    }

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
      timeline,
    };
  }
}
