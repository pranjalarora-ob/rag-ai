import { Injectable } from '@nestjs/common';
import { QdrantService } from './qdrant.service';
import { COLLECTION } from './constants';
import { toRegion } from './region.util';

// Cap on rows returned for list operations — keeps responses fast/cheap.
const MAX_ROWS = 30;

// Milestones/phases carry a status of COMPLETED | IN_PROGRESS | null (not started).
// A milestone's `endDate` is its DUE DATE (target), so date questions ("due in N
// days", "overdue", "behind schedule") ARE answerable off endDate + status.
type StatusFilter = 'COMPLETED' | 'IN_PROGRESS' | 'PENDING' | 'NOT_STARTED';

export interface FlowQuery {
  customerId: string;
  // What to return: count of matching projects, list them, list matching milestones,
  // or a funnel grouped by each project's current phase.
  operation?: 'countProjects' | 'listProjects' | 'listMilestones' | 'funnel';
  projectCode?: string;
  // Phase-level filters (phase names: Pre-Sales, Design, Execution, Handover, ...).
  phaseName?: string; // substring match
  phaseStatus?: StatusFilter;
  // Milestone-level filters (e.g. name "Payment", "Handover", "Mobilization").
  milestoneName?: string; // substring match
  milestoneStatus?: StatusFilter;
  // Cross-milestone condition: a project that has completed milestone A but NOT B.
  // e.g. hasCompleted="Mobilization Advance", missing="Site Kick-Off".
  hasCompleted?: string; // substring — milestone completed
  missing?: string; // substring — milestone NOT completed
  // Due-date filters — endDate is treated as the milestone due date. These implicitly
  // exclude COMPLETED milestones (a done milestone isn't "due"/"overdue") unless an
  // explicit milestoneStatus is also passed.
  overdue?: boolean; // due date in the past and not completed
  dueWithinDays?: number; // due date between now and now + N days
  dueBefore?: string; // ISO date — due on/before this
  dueAfter?: string; // ISO date — due on/after this
  // ---- project-attribute join (flow docs carry no zone/city/owner) ----
  zone?: string; // filter matched projects by raw zone
  region?: string; // filter by normalized region (North/South/East/West)
  city?: string; // filter matched projects by city
  owner?: string; // filter matched projects by owner
  // Group the matched projects by a project attribute or their current phase.
  groupBy?: 'region' | 'zone' | 'city' | 'owner' | 'currentPhase';
}

interface ProjectAttr {
  zone?: string;
  region: string;
  city?: string;
  owner?: string;
}

interface FlowPhase {
  name: string;
  code?: string;
  status: string | null;
  sequence: number;
  startDate: string | null;
  endDate: string | null;
  milestones: FlowMilestone[];
}
interface FlowMilestone {
  name: string;
  code?: string;
  status: string | null;
  sequence: number;
  startDate: string | null;
  endDate: string | null;
}
interface FlowProject {
  projectCode: string;
  projectName: string;
  phases: FlowPhase[];
}

/**
 * Deterministic query engine over project-flow-phase docs (phases + nested milestones).
 * Answers status-based workflow questions across ALL of a customer's projects —
 * "which projects have a pending milestone payment", "projects at handover stage",
 * "mobilization advance received but site not started", stage funnel — with exact
 * counts (no LLM math). Dates are actuals only; there is no due-date field, so this
 * engine intentionally does not attempt "due in N days" / "overdue" questions.
 */
@Injectable()
export class ProjectFlowService {
  constructor(private readonly qdrant: QdrantService) {}

  private up(s: any): string {
    return String(s ?? '').trim().toUpperCase();
  }

  // Match an actual status against a requested filter. PENDING = anything not
  // completed (null or in-progress) — the common business meaning of "pending".
  private statusMatch(actual: string | null, want?: StatusFilter): boolean {
    if (!want) return true;
    const a = this.up(actual);
    switch (want) {
      case 'COMPLETED':
        return a === 'COMPLETED';
      case 'IN_PROGRESS':
        return a === 'IN_PROGRESS';
      case 'NOT_STARTED':
        return a === '' ;
      case 'PENDING':
      default:
        return a !== 'COMPLETED';
    }
  }

  private includes(haystack: any, needle?: string): boolean {
    if (!needle) return true;
    return String(haystack ?? '').toLowerCase().includes(needle.trim().toLowerCase());
  }

  private parseDate(s: any): Date | null {
    if (!s) return null;
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }

  // Does a milestone satisfy the name + status + due-date conditions? `endDate` is the
  // due date. Date conditions default to excluding COMPLETED milestones unless the
  // caller passes an explicit milestoneStatus.
  private milestonePredicate(
    m: { name: string; status: string | null; endDate: string | null },
    q: FlowQuery,
    now: Date,
  ): boolean {
    if (!this.includes(m.name, q.milestoneName)) return false;

    const hasDateFilter = q.overdue || q.dueWithinDays != null || q.dueBefore || q.dueAfter;
    if (hasDateFilter) {
      const due = this.parseDate(m.endDate);
      if (!due) return false; // no due date recorded → can't match a date condition
      if (q.overdue) {
        if (due >= now) return false;
        if (q.milestoneStatus == null && this.up(m.status) === 'COMPLETED') return false;
      }
      if (q.dueWithinDays != null) {
        const limit = new Date(now.getTime() + q.dueWithinDays * 86400000);
        if (due < now || due > limit) return false;
        if (q.milestoneStatus == null && this.up(m.status) === 'COMPLETED') return false;
      }
      if (q.dueAfter) {
        const after = this.parseDate(q.dueAfter);
        if (after && due < after) return false;
      }
      if (q.dueBefore) {
        const before = this.parseDate(q.dueBefore);
        if (before && due > before) return false;
      }
    }

    if (q.milestoneStatus && !this.statusMatch(m.status, q.milestoneStatus)) return false;
    return true;
  }

  // Load all flow-phase docs for a customer, deduped by phaseId, grouped per project.
  private async loadProjects(customerId: string, projectCode?: string): Promise<FlowProject[]> {
    const must: any[] = [
      { key: 'customerId', match: { value: customerId } },
      { key: 'docType', match: { value: 'project-flow-phase' } },
    ];
    if (projectCode) must.push({ key: 'projectCode', match: { value: String(projectCode) } });

    const points = await this.qdrant.scrollAll(COLLECTION, { must });
    const seenPhase = new Set<string>();
    const byProject = new Map<string, FlowProject>();

    for (const pt of points) {
      const pl: any = pt.payload || {};
      const phaseId = pl.phaseId || pl.original_id;
      if (phaseId && seenPhase.has(phaseId)) continue;
      if (phaseId) seenPhase.add(phaseId);

      const code = String(pl.projectCode ?? '');
      if (!code) continue;
      let proj = byProject.get(code);
      if (!proj) {
        proj = { projectCode: code, projectName: pl.projectName || `Project ${code}`, phases: [] };
        byProject.set(code, proj);
      }
      proj.phases.push({
        name: pl.name,
        code: pl.code,
        status: pl.status ?? null,
        sequence: Number(pl.sequence) || 0,
        startDate: pl.startDate ?? null,
        endDate: pl.endDate ?? null,
        milestones: (Array.isArray(pl.milestones) ? pl.milestones : []).map((m: any) => ({
          name: m.name,
          code: m.code,
          status: m.status ?? null,
          sequence: Number(m.sequence) || 0,
          startDate: m.startDate ?? null,
          endDate: m.endDate ?? null,
        })),
      });
    }

    for (const proj of byProject.values()) {
      proj.phases.sort((a, b) => a.sequence - b.sequence);
    }
    return [...byProject.values()];
  }

  // Flow docs carry no zone/city/owner, so load those from the project docs and key
  // them by projectCode — used to filter/group flow matches by project attributes.
  private async loadProjectAttrs(customerId: string): Promise<Map<string, ProjectAttr>> {
    const points = await this.qdrant.scrollAll(COLLECTION, {
      must: [
        { key: 'customerId', match: { value: customerId } },
        { key: 'docType', match: { value: 'project' } },
      ],
    });
    const map = new Map<string, ProjectAttr>();
    for (const pt of points) {
      const p: any = pt.payload || {};
      const code = String(p.projectCode ?? '');
      if (!code || map.has(code)) continue;
      map.set(code, { zone: p.zone, region: toRegion(p.zone), city: p.city, owner: p.owner });
    }
    return map;
  }

  // A project's current position in the flow: the in-progress phase, else the first
  // phase not yet completed, else the last phase. Drives the funnel view.
  private currentPhase(proj: FlowProject): string {
    const inProg = proj.phases.find((p) => this.up(p.status) === 'IN_PROGRESS');
    if (inProg) return inProg.name;
    const firstOpen = proj.phases.find((p) => this.up(p.status) !== 'COMPLETED');
    return (firstOpen || proj.phases[proj.phases.length - 1])?.name || 'Unknown';
  }

  // Flatten a project's milestones (optionally scoped to a phase) into rows, each
  // tagged with its phase — used for name/status matching and listMilestones output.
  private milestonesOf(proj: FlowProject, phaseName?: string) {
    const rows: Array<{ phase: string; name: string; status: string | null; startDate: string | null; endDate: string | null }> = [];
    for (const ph of proj.phases) {
      if (phaseName && !this.includes(ph.name, phaseName)) continue;
      for (const m of ph.milestones) {
        rows.push({ phase: ph.name, name: m.name, status: m.status, startDate: m.startDate, endDate: m.endDate });
      }
    }
    return rows;
  }

  async analyze(q: FlowQuery) {
    const projects = await this.loadProjects(q.customerId, q.projectCode);
    const operation = q.operation || 'listProjects';
    const now = new Date();

    // Project-attribute join: flow docs carry no zone/city/owner, so load them from
    // the project docs when a filter or grouping needs them.
    const attrFilterActive = !!(q.zone || q.region || q.city || q.owner);
    const needAttrs = attrFilterActive || (!!q.groupBy && q.groupBy !== 'currentPhase');
    const attrs = needAttrs ? await this.loadProjectAttrs(q.customerId) : null;

    const passAttr = (code: string): boolean => {
      if (!attrFilterActive) return true;
      const a = attrs?.get(code);
      if (q.zone && !this.includes(a?.zone, q.zone)) return false;
      if (q.city && String(a?.city || '').toLowerCase() !== q.city.trim().toLowerCase()) return false;
      if (q.region && String(a?.region || '').toLowerCase() !== q.region.trim().toLowerCase()) return false;
      if (q.owner && !this.includes(a?.owner, q.owner)) return false;
      return true;
    };
    const groupKey = (code: string, phase: string): string => {
      if (q.groupBy === 'currentPhase') return phase;
      const a = attrs?.get(code) as any;
      return String((a && a[q.groupBy!]) ?? 'Unknown');
    };

    // Funnel: group projects by their current phase (workflow-stage counts).
    if (operation === 'funnel') {
      const breakdown: Record<string, number> = {};
      let total = 0;
      for (const proj of projects) {
        if (!passAttr(proj.projectCode)) continue;
        total += 1;
        const key = this.currentPhase(proj);
        breakdown[key] = (breakdown[key] || 0) + 1;
      }
      return { operation, totalProjects: total, breakdown };
    }

    // Evaluate the per-project match predicate from the supplied filters.
    const matches: Array<{ projectCode: string; projectName: string; currentPhase: string; matched: any[] }> = [];
    for (const proj of projects) {
      if (!passAttr(proj.projectCode)) continue;
      // Cross-milestone condition (has A completed, B not completed).
      if (q.hasCompleted || q.missing) {
        const all = this.milestonesOf(proj);
        const okHas = !q.hasCompleted || all.some((m) => this.includes(m.name, q.hasCompleted) && this.up(m.status) === 'COMPLETED');
        const okMissing = !q.missing || all.some((m) => this.includes(m.name, q.missing) && this.up(m.status) !== 'COMPLETED');
        if (!okHas || !okMissing) continue;
      }

      // Phase-level filter.
      if (q.phaseName || q.phaseStatus) {
        const phaseHit = proj.phases.some(
          (ph) => this.includes(ph.name, q.phaseName) && this.statusMatch(ph.status, q.phaseStatus),
        );
        if (!phaseHit) continue;
      }

      // Milestone-level filter — collect the specific milestones that matched
      // (name + status + due-date conditions, endDate = due date).
      let matchedMs = this.milestonesOf(proj, q.phaseName);
      const wantMsFilter =
        q.milestoneName || q.milestoneStatus || q.overdue || q.dueWithinDays != null || q.dueBefore || q.dueAfter;
      if (wantMsFilter) {
        matchedMs = matchedMs.filter((m) => this.milestonePredicate(m, q, now));
        if (!matchedMs.length) continue;
      }

      matches.push({
        projectCode: proj.projectCode,
        projectName: proj.projectName,
        currentPhase: this.currentPhase(proj),
        matched: matchedMs.slice(0, 8),
      });
    }

    // Grouped counts by project attribute (region/zone/city/owner) or current phase —
    // e.g. "zone-wise count of projects pending client milestone payment".
    if (q.groupBy) {
      const breakdown: Record<string, number> = {};
      for (const m of matches) {
        const key = groupKey(m.projectCode, m.currentPhase);
        breakdown[key] = (breakdown[key] || 0) + 1;
      }
      return { operation: 'groupBy', groupBy: q.groupBy, count: matches.length, breakdown };
    }

    if (operation === 'countProjects') {
      return { operation, count: matches.length };
    }

    if (operation === 'listMilestones') {
      const rows = matches
        .flatMap((p) => p.matched.map((m) => ({ projectCode: p.projectCode, projectName: p.projectName, ...m })))
        .slice(0, MAX_ROWS);
      return { operation, count: matches.length, shown: rows.length, rows };
    }

    // listProjects (default)
    return {
      operation,
      count: matches.length,
      shown: Math.min(matches.length, MAX_ROWS),
      rows: matches.slice(0, MAX_ROWS).map((m) => ({
        projectCode: m.projectCode,
        projectName: m.projectName,
        currentPhase: m.currentPhase,
        matchedMilestones: m.matched.map((x) => `${x.name} [${x.status || 'NOT_STARTED'}]`),
      })),
    };
  }
}
