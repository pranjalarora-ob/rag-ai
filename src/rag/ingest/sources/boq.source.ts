import { COLLECTION } from '../../constants';
import { IngestDocument, SourceDefinition } from '../source.types';

/**
 * Source: bs_boqs (boq-service DB, camelCase timestamp columns -> quoted + aliased).
 *
 * Folded children (important ones only):
 *   - rfqs          : bs_rfq        (summary: name/status/vendor/cost)
 *   - workitemCount : bs_boq_workitem count (line items are NOT embedded individually)
 *
 * Linked to a project via projectId (= bs_boqs.project_id = ls_lead_projects.id).
 * ADD A NEW CHILD LATER → add a sub-select in select() + a line in toDocument().
 */

const select = (where: string) => `
SELECT
  t.id                AS id,
  t.account_id        AS account_id,
  t.project_id        AS project_id,
  t.name, t.code, t.type, t.status, t.stage,
  t.area, t.height,
  t.cost, t.buy_value, t.sales_cost, t.discounted_cost,
  t.validity_date,
  t."createdAt"       AS created_at,
  t."updatedAt"       AS updated_at,

  COALESCE((SELECT json_agg(json_build_object(
     'name', r.name, 'status', r.status, 'cost', r.cost, 'vendor', r.vendor_name))
   FROM bs_rfqs r WHERE r.boq_id = t.id AND r."deletedAt" IS NULL), '[]') AS rfqs,

  (SELECT COUNT(*) FROM bs_boq_workitems wi
   WHERE wi.boq_id = t.id AND wi."deletedAt" IS NULL) AS workitem_count,

  -- Fold the parent project's identity in so a BOQ doc is self-contained
  -- (answers "which project is this BOQ for" without a cross-doc join).
  p.name           AS project_name,
  p.company_name   AS project_company,
  p.city           AS project_city,
  p.owner          AS project_owner,
  p.code           AS project_code
FROM bs_boqs t
LEFT JOIN ls_lead_projects p ON p.id = t.project_id
${where}`;

const num = (x: any) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
};

export const boqSource: SourceDefinition = {
  name: 'boq',
  db: 'boq',
  collection: COLLECTION,
  pageSql: select(
    `WHERE t."deletedAt" IS NULL AND t."updatedAt" >= $1 AND t.id > $2 ORDER BY t.id ASC LIMIT $3`,
  ),
  byIdSql: select(`WHERE t.id = $1`),

  toDocument(row: any): IngestDocument {
    const rfqs: any[] = Array.isArray(row.rfqs) ? row.rfqs : [];

    const lines = [
      `BOQ ${row.name || ''} (code ${row.code || ''}, type ${row.type || ''}).`,
      `Project: ${(row.project_company || '').trim()}${row.project_name ? ` — ${row.project_name}` : ''} (City: ${row.project_city || ''}, Owner: ${row.project_owner || ''}).`,
      `Status: ${row.status}. Stage: ${row.stage}.`,
      `Area: ${row.area}. Height: ${row.height}.`,
      `Cost: ${row.cost}. Buy value: ${row.buy_value}. Sales cost: ${row.sales_cost}. Discounted cost: ${row.discounted_cost}.`,
      `Work items: ${row.workitem_count}. RFQs: ${rfqs
        .map((r) => `${r.name || ''} [${r.status || ''}] ${r.vendor || ''} ₹${r.cost || 0}`)
        .join('; ')}.`,
    ];

    return {
      id: row.id,
      text: lines.join('\n'),
      metadata: {
        docType: 'boq',
        boqId: row.id,
        projectId: row.project_id,
        projectName: row.project_name,
        projectCode: row.project_code ? String(row.project_code) : null,
        companyName: row.project_company,
        city: row.project_city,
        owner: row.project_owner,
        accountId: row.account_id,
        customerId: row.account_id, // compat with existing customerId filter
        boqCode: row.code,
        boqType: row.type,
        status: row.status,
        stage: row.stage,
        cost: num(row.cost),
        buyValue: num(row.buy_value),
        salesCost: num(row.sales_cost),
        workitemCount: num(row.workitem_count),
        rfqCount: rfqs.length,
        updatedAt: row.updated_at,
      },
    };
  },
};
