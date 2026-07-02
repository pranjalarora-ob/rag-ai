import { COLLECTION } from '../../constants';
import { IngestDocument, SourceDefinition } from '../source.types';

/**
 * Source: ls_lead_projects (lead-service DB, snake_case columns).
 *
 * Folded children (important ones only, per current query needs):
 *   - customer        : from t.customer_info JSONB (no fragile cross-table FK)
 *   - team            : ls_assigned_resources  (powers "projects where X is a team member")
 *   - commercial (1:1): ls_project_commercial_summaries
 *   - mis (1:1)       : ls_project_mis_summaries
 *
 * ADD A NEW CHILD LATER → add a sub-select in select() + a line in toDocument().
 */

const select = (where: string) => `
SELECT
  t.id                       AS id,
  t.account_id               AS account_id,
  t.code, t.name, t.company_name, t.nature_of_business,
  t.city, t.state, t.zone, t.area_sft,
  t.type, t.stage, t.sub_stage, t.priority, t.lead_type,
  t.lead_status, t.project_status, t.owner, t.scope, t.channel,
  t.estimated_value, t.current_project_value, t.closure_value,
  t.customer_info,
  t."desc"                   AS descr,
  t."createdAt"              AS created_at,
  t."updatedAt"              AS updated_at,

  (SELECT json_build_object(
     'totalProjectValue', cs.total_project_value,
     'receivedAmount',    cs.received_amount,
     'balanceAmount',     cs.balance_amount,
     'agreedHandover',    cs.agreed_handover_date_contract,
     'keyRisks',          cs.key_risks)
   FROM ls_project_commercial_summaries cs
   WHERE cs.project_id = t.id AND cs."deletedAt" IS NULL
   LIMIT 1)                  AS commercial,

  (SELECT json_build_object(
     'projectValueWoTax', mis.current_project_value_wo_tax,
     'designOrderValue',  mis.current_design_order_value,
     'buildOrderValue',   mis.tbc_build_order_value,
     'cogs',              mis.cogs,
     'cogsPercentage',    mis.cogs_percentage,
     'leadSource',        mis.lead_source)
   FROM ls_project_mis_summaries mis
   WHERE mis.project_id = t.id
   LIMIT 1)                  AS mis,

  COALESCE((SELECT json_agg(json_build_object(
     'userId', ar.user_id, 'role', ar.role, 'pocRole', ar.poc_role))
   FROM ls_assigned_resources ar
   WHERE ar.project_id = t.id AND ar."deletedAt" IS NULL), '[]') AS team

FROM ls_lead_projects t
${where}`;

const num = (x: any) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
};
const stripHtml = (s: any) =>
  String(s || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

export const projectSource: SourceDefinition = {
  name: 'project',
  db: 'lead',
  collection: COLLECTION,
  pageSql: select(
    `WHERE t."deletedAt" IS NULL AND t."updatedAt" >= $1 AND t.id > $2 ORDER BY t.id ASC LIMIT $3`,
  ),
  byIdSql: select(`WHERE t.id = $1`),

  toDocument(row: any): IngestDocument {
    const cust = row.customer_info || {};
    const com = row.commercial || {};
    const mis = row.mis || {};
    const team: any[] = Array.isArray(row.team) ? row.team : [];

    const lines = [
      `Project ${row.code ?? ''} (${(row.company_name || '').trim()}).`,
      `Type: ${row.lead_type}. Lead status: ${row.lead_status}. Project status: ${row.project_status}.`,
      `Customer: ${cust.name || cust.customerName || ''} (${cust.emailId || cust.email || ''}, ${cust.mobileNumber || cust.mobile_number || ''}).`,
      `City: ${row.city}, ${row.state}, zone ${row.zone}. Area: ${row.area_sft} sqft. Nature of business: ${row.nature_of_business || ''}.`,
      `Stage: ${row.stage} / ${row.sub_stage}. Priority: ${row.priority}. Owner: ${row.owner}. Scope: ${row.scope}. Channel: ${row.channel}.`,
      `Estimated value: ${row.estimated_value}. Current project value: ${row.current_project_value}. Closure value: ${row.closure_value}.`,
      `Team: ${team.map((m) => `${m.userId} (${m.role || m.pocRole || ''})`).join(', ')}.`,
      `Notes: ${stripHtml(row.descr)}.`,
    ];
    if (com && Object.keys(com).length) {
      lines.push(
        `Commercial: total value ${com.totalProjectValue}, received ${com.receivedAmount}, balance ${com.balanceAmount}, handover ${com.agreedHandover}. Key risks: ${com.keyRisks || 'none'}.`,
      );
    }
    if (mis && Object.keys(mis).length) {
      lines.push(
        `MIS: value w/o tax ${mis.projectValueWoTax}, design order ${mis.designOrderValue}, build order ${mis.buildOrderValue}, COGS ${mis.cogs} (${mis.cogsPercentage}%), lead source ${mis.leadSource}.`,
      );
    }

    return {
      id: row.id,
      text: lines.join('\n'),
      metadata: {
        docType: 'project',
        type: row.type ? String(row.type).toLowerCase() : null, // DB type LEAD|PROJECT -> "lead"|"project"
        projectId: row.id,
        accountId: row.account_id,
        customerId: row.account_id, // keeps the existing chat/analytics customerId filter working
        projectCode: row.code ? String(row.code) : null,
        projectName: row.name,
        companyName: row.company_name,
        customerInfo: {
          name: cust.name || cust.customerName || null,
          email: cust.emailId || cust.email || null,
          mobile: cust.mobileNumber || cust.mobile_number || null,
        },
        city: row.city,
        state: row.state,
        zone: row.zone,
        areaSft: num(row.area_sft),
        channel: row.channel,
        stage: row.stage,
        subStage: row.sub_stage,
        leadStatus: row.lead_status,
        projectStatus: row.project_status,
        owner: row.owner,
        estimatedValue: num(row.estimated_value),
        currentProjectValue: num(row.current_project_value),
        closureValue: num(row.closure_value),
        team, // [{ userId, role, pocRole }] — filterable for "where <person> is a team member"
        updatedAt: row.updated_at,
      },
    };
  },
};
