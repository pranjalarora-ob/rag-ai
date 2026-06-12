import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { COLLECTION } from './constants';

type Metric = 'estimatedValue' | 'boqValue';

export interface AnalyticsQuery {
  customerId: string;
  metric?: Metric;          // money field to aggregate (default estimatedValue)
  topN?: number;            // how many top projects to return (default 5)
  teamMember?: string;      // filter to projects where this person is on the team
  role?: string;            // optional role to match alongside teamMember
  groupBy?: 'stage' | 'city' | 'owner';
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

  private async loadProjects(customerId: string) {
    const points = await this.qdrantService.scrollAll(COLLECTION, {
      must: [{ key: 'customerId', match: { value: customerId } }],
    });
    return points.map((p) => p.payload || {});
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
    const topN = query.topN ?? 5;

    let projects = await this.loadProjects(query.customerId);

    if (query.teamMember) {
      projects = projects.filter((p) => this.matchesTeam(p, query.teamMember!, query.role));
    }

    const rows = projects.map((p) => ({
      projectCode: p.projectCode,
      companyName: (p.companyName || '').trim(),
      city: p.city,
      stage: p.stage,
      owner: p.owner,
      value: this.num(p[metric]),
    }));

    const total = rows.reduce((sum, r) => sum + r.value, 0);
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
      filter: query.teamMember ? { teamMember: query.teamMember, role: query.role || 'any' } : null,
      count: rows.length,
      total,
      average: rows.length ? total / rows.length : 0,
      top,
      breakdown,
    };
  }
}
