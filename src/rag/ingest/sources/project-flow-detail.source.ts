import { COLLECTION } from '../../constants';
import { IngestDocument, SourceDefinition } from '../source.types';

const select = (where: string) => `
SELECT
  t.id                AS id,
  t.project_id        AS project_id,
  t.name,
  t.status,
  t.code,
  t.sequence,
  t.start_date,
  t.end_date,
  t."createdAt"       AS created_at,
  t."updatedAt"       AS updated_at,
  
  -- Join parent project identity and customer ID (account_id)
  p.account_id        AS account_id,
  p.name              AS project_name,
  p.code              AS project_code,
  p.company_name      AS project_company,

  -- Subquery to aggregate all child milestones/tasks for this phase
  COALESCE((
    SELECT json_agg(json_build_object(
      'id', c.id,
      'name', c.name,
      'status', c.status,
      'code', c.code,
      'sequence', c.sequence,
      'startDate', c.start_date,
      'endDate', c.end_date
    ) ORDER BY c.sequence ASC)
    FROM ls_project_flow_details c
    WHERE c.parent_id = t.id AND c."deletedAt" IS NULL
  ), '[]') AS milestones

FROM ls_project_flow_details t
JOIN ls_lead_projects p ON p.id = t.project_id
${where}`;

const formatDate = (d: any) => {
  if (!d) return '';
  const date = new Date(d);
  return isNaN(date.getTime()) ? String(d) : date.toISOString().split('T')[0];
};

export const projectFlowDetailSource: SourceDefinition = {
  name: 'project-flow-detail',
  db: 'lead',
  collection: COLLECTION,
  pageSql: select(
    `WHERE t.parent_id IS NULL AND t."deletedAt" IS NULL AND t."updatedAt" >= $1 AND t.id > $2 ORDER BY t.id ASC LIMIT $3`,
  ),
  byIdSql: select(`WHERE t.id = $1 AND t.parent_id IS NULL AND t."deletedAt" IS NULL`),

  toDocument(row: any): IngestDocument {
    const milestones = Array.isArray(row.milestones) ? row.milestones : [];

    // Calculate overall start/end dates from the phase's dates and all milestones' dates
    let overallStart = row.start_date ? new Date(row.start_date) : null;
    if (overallStart && isNaN(overallStart.getTime())) overallStart = null;

    let overallEnd = row.end_date ? new Date(row.end_date) : null;
    if (overallEnd && isNaN(overallEnd.getTime())) overallEnd = null;

    for (const m of milestones) {
      if (m.startDate) {
        const d = new Date(m.startDate);
        if (!isNaN(d.getTime())) {
          if (!overallStart || d < overallStart) {
            overallStart = d;
          }
        }
      }
      if (m.endDate) {
        const d = new Date(m.endDate);
        if (!isNaN(d.getTime())) {
          if (!overallEnd || d > overallEnd) {
            overallEnd = d;
          }
        }
      }
    }

    const start = overallStart ? formatDate(overallStart) : '';
    const end = overallEnd ? formatDate(overallEnd) : '';
    const dateStr = [start ? `Start: ${start}` : '', end ? `End: ${end}` : ''].filter(Boolean).join(', ');

    const lines = [
      `Project Workflow Phase: ${row.name || ''} [Code: ${row.code || 'N/A'}] (Status: ${row.status || 'Pending'}, Sequence: ${row.sequence || 0})${dateStr ? ` (${dateStr})` : ''}.`,
      `Associated Project: ${(row.project_company || '').trim()}${row.project_name ? ` — ${row.project_name}` : ''} (Code: ${row.project_code || ''}).`,
    ];

    if (milestones.length > 0) {
      lines.push('Tasks/Milestones in this Phase:');
      for (const m of milestones) {
        const mStart = formatDate(m.startDate);
        const mEnd = formatDate(m.endDate);
        const mDates = [mStart ? `Start: ${mStart}` : '', mEnd ? `End: ${mEnd}` : ''].filter(Boolean).join(', ');

        lines.push(`  * Milestone: ${m.name} [Code: ${m.code || 'N/A'}] (Status: ${m.status || 'Pending'}, Sequence: ${m.sequence || 0})${mDates ? ` (${mDates})` : ''}`);
      }
    }

    return {
      id: row.id,
      text: lines.join('\n'),
      metadata: {
        docType: 'project-flow-phase', // <-- Represents a phase document
        phaseId: row.id,
        projectId: row.project_id,
        projectName: row.project_name,
        projectCode: row.project_code ? String(row.project_code) : null,
        parentId: row.parent_id,
        customerId: row.account_id, // Critical: Keeps search security working!
        name: row.name,
        status: row.status,
        code: row.code,
        sequence: Number(row.sequence) || 0,
        startDate: start || null,
        endDate: end || null,
        milestones: milestones.map((m: any) => ({
          id: m.id,
          name: m.name,
          status: m.status,
          code: m.code,
          sequence: Number(m.sequence) || 0,
          startDate: m.startDate ? formatDate(m.startDate) : null,
          endDate: m.endDate ? formatDate(m.endDate) : null,
        })),
        updatedAt: row.updated_at,
      },
    };
  },
};
