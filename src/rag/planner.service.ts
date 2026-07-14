import { Injectable } from '@nestjs/common';
import { Response } from 'express';
import { OpenaiService } from './openai.service';
import { QdrantService } from './qdrant.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { ProjectFlowService } from './project-flow.service';
import { RerankService } from './rerank.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

const MAX_STEPS = 5; // safety cap on the tool→evaluate→tool loop

// Appended to the planner's system prompt so list-style answers render as tables
// the UI can turn into charts. The chart parser needs a "Name" column plus a
// "Value"/"Area" column, so the columns below must stay in sync with it.
const TABLE_FORMAT_INSTRUCTION = `
=== CRITICAL OUTPUT RULES — MUST FOLLOW EXACTLY ===

RULE 1 — NEVER say you cannot draw a chart. The UI draws charts FOR you from tables.
RULE 2 — NEVER use bullet points or numbered lists when showing multiple projects.
RULE 3 — ALWAYS use a markdown table when showing 2+ projects. Exact columns:
  | # | Code | Name | City | Area (sqft) | Estimated Value |
  Use RAW integers in Area and Estimated Value (NO commas, NO "Cr"/"L" suffix, NO units).
  COLUMN MAPPING (STRICT): take the "Area (sqft)" column from each row's "area" field
  and the "Estimated Value" column from each row's "estimatedValue" field. NEVER use the
  ranking "value" field for the Estimated Value column — when ranking by area, "value"
  equals the area and must NOT be repeated as the estimated value. Area and Estimated
  Value are different numbers; never output the same number in both columns.

RULE 4 — When the user asks for a "chart", "graph", "pie", "bar", "comparison", or "visualize":
  → Call projectAnalytics with metric="area" (or relevant metric) and topN=50
  → Return the results as the markdown table above
  → Do NOT say anything like "I am unable to provide a chart"

Example of CORRECT response to "show top 5 projects by area":
| # | Code | Name | City | Area (sqft) | Estimated Value |
|---|------|------|------|-------------|-----------------|
| 1 | P001 | Alpha Tower | Delhi | 85000 | 50000000 |
| 2 | P002 | Beta Mall | Mumbai | 72000 | 43000000 |

RULE 5 — Only use plain prose (no table) for single facts, yes/no, or counts with no row data.
=== END CRITICAL RULES ===`;

export interface PlannerResult {
  answer: string;
  trace: Array<{ tool: string; args: any; result: any }>;
  steps: number;
}

/**
 * The PLANNER ("thinker"): an LLM tool-calling loop. It decides which tool(s) to call
 * (searchProjects = RAG lookup, projectAnalytics = exact math), runs them, reads the
 * results, and drafts the final answer. Generation via OpenRouter (OpenAI tool format).
 */
@Injectable()
export class PlannerService {
  constructor(
    private readonly openaiService: OpenaiService,
    private readonly qdrantService: QdrantService,
    private readonly projectAnalyticsService: ProjectAnalyticsService,
    private readonly projectFlowService: ProjectFlowService,
    private readonly rerankService: RerankService,
  ) { }

  private readonly tools = [
    {
      type: 'function',
      function: {
        name: 'searchProjects',
        description:
          'Search project records by meaning. Use for DESCRIPTIVE lookups about a specific project — who is the design manager, what stage, details/notes. Returns matching project summaries.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'A natural-language search query.' },
          },
          required: ['query'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'projectAnalytics',
        description:
          'Compute EXACT aggregates over ALL of the customer\'s projects: totals, averages, counts, top-N rankings, filtered sums/lists. Use for any "total", "how many", "top N", "highest/lowest", "average", or "where <person> is <role>" question. Never compute numbers yourself — call this.',
        parameters: {
          type: 'object',
          properties: {
            metric: { type: 'string', enum: ['estimatedValue', 'boqValue', 'area'], description: 'What to rank/total/chart by. Use "area" for area-based questions ("compare their area", "largest by area").' },
            topN: { type: 'number', description: 'Return only the top N by value/area. Set to 50 when asked to chart/compare all results or when a previous list was longer than 5.' },
            projectCodes: { type: 'array', items: { type: 'string' }, description: 'Look up SPECIFIC projects by their exact project code(s), e.g. ["2022072479"]. ALWAYS use this when the user names one or more project codes — never search by meaning for a code.' },
            teamMember: { type: 'string', description: 'Filter to projects where this person is on the team.' },
            role: { type: 'string', description: 'Role to match with teamMember, e.g. "Design Manager".' },
            groupBy: {
              type: 'string',
              enum: ['stage', 'subStage', 'city', 'state', 'region', 'owner', 'channel', 'projectStatus'],
              description: 'Group counts/totals by this field, e.g. "how many projects per stage". For any zone-wise / region-wise question ALWAYS use "region" (North/South/East/West) — never group by raw zone.',
            },
            groupBy2: {
              type: 'string',
              enum: ['stage', 'subStage', 'city', 'state', 'region', 'owner', 'channel', 'projectStatus'],
              description: 'SECOND dimension for a cross-tab/pivot. For "X-wise breakdown BY Y" set groupBy=X and groupBy2=Y (e.g. "city-wise breakdown by stage" → groupBy="city", groupBy2="stage"). The result "pivot" is X→Y→count; render it as a table with X as rows and Y as columns.',
            },
            groupByRole: {
              type: 'string',
              description: 'Group by the PERSON holding a team role. Use "General Manager" for any "per GM / GM workload / which GM" question (a GM is the team member with role General Manager). Returns a breakdown keyed by each person\'s name.',
            },
            city: { type: 'string', description: 'Filter by city, e.g. "Gurugram".' },
            state: { type: 'string', description: 'Filter by state, e.g. "Haryana".' },
            zone: { type: 'string', description: 'Filter by raw zone/hub, e.g. "Gurgaon".' },
            region: { type: 'string', description: 'Filter by normalized region: North, South, East, or West. Prefer this over zone for "in the North" style questions.' },
            stage: { type: 'string', description: 'Filter by stage, e.g. "Execution", "Design-Sales".' },
            subStage: { type: 'string', description: 'Filter by sub-stage, e.g. "Handover".' },
            owner: { type: 'string', description: 'Filter by owner/team, e.g. "SMB Team".' },
            channel: { type: 'string', description: 'Filter by channel, e.g. "Digital".' },
            projectStatus: { type: 'string', description: 'Filter by project status, e.g. "Cancelled", "InProgress".' },
            companyName: { type: 'string', description: 'Filter by company/project company name, e.g. "Officebanao".' },
            customerName: { type: 'string', description: 'Filter by the customer contact name.' },
            minValue: { type: 'number', description: 'estimatedValue >= this (in rupees). Convert "2 cr" -> 20000000, "40 lakh" -> 4000000.' },
            maxValue: { type: 'number', description: 'estimatedValue <= this (in rupees).' },
            equalsValue: { type: 'number', description: 'estimatedValue EXACTLY equals this (in rupees). Use for "equal to"/"= X" queries. Convert "1.65L" -> 165000, "2 cr" -> 20000000. Takes precedence over min/maxValue.' },
            minArea: { type: 'number', description: 'areaSft >= this (in sqft).' },
            maxArea: { type: 'number', description: 'areaSft <= this (in sqft).' },
            equalsArea: { type: 'number', description: 'areaSft EXACTLY equals this (in sqft). Use for "area equal to X". Takes precedence over min/maxArea.' },
            type: { type: 'string', enum: ['lead', 'project'], description: 'Filter by record type: "lead" (not yet converted) or "project". Use for "how many leads..." vs "how many projects...".' },
            active: { type: 'boolean', description: 'true = only active projects, false = only inactive. Use for "active projects".' },
            ownerMissing: { type: 'boolean', description: 'true = only records with NO assigned owner. Use for "unassigned" / "no owner".' },
            dateField: { type: 'string', enum: ['createdAt', 'updatedAt'], description: 'Which date the date filters/buckets use. Default createdAt (creation/lead date).' },
            lastNDays: { type: 'number', description: 'Only records whose date is within the last N days. Use for "last 30 days".' },
            lastNMonths: { type: 'number', description: 'Only records whose date is within the last N months. Use for "last 3 months", "last 6 months".' },
            createdAfter: { type: 'string', description: 'ISO date (YYYY-MM-DD). Only records dated on/after this.' },
            createdBefore: { type: 'string', description: 'ISO date (YYYY-MM-DD). Only records dated on/before this.' },
            bucketBy: { type: 'string', enum: ['month', 'quarter'], description: 'Return a timeline breakdown grouped by month or quarter. Use for trends ("trend over last 6 months", "this quarter vs last").' },
          },
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'queryProjectFlow',
        description:
          'Query the project WORKFLOW/STAGE data (phases + milestones with status and due dates) across all projects. Use for questions about milestone status, phase progress, pending payments, handover stage, "stuck at" a stage, the stage funnel, and DUE-DATE questions — overdue milestones, "due in the next N days", behind schedule. A milestone\'s endDate is its due date. Status is COMPLETED / IN_PROGRESS / PENDING (not completed).',
        parameters: {
          type: 'object',
          properties: {
            operation: {
              type: 'string',
              enum: ['countProjects', 'listProjects', 'listMilestones', 'funnel'],
              description: 'countProjects = how many projects match; listProjects = list them; listMilestones = list the matching milestones; funnel = count of projects grouped by their current phase (stage funnel view).',
            },
            projectCode: { type: 'string', description: 'Restrict to a single project by its code.' },
            phaseName: { type: 'string', description: 'Filter by phase name (substring), e.g. "Execution", "Design", "Handover", "Pre-Sales".' },
            phaseStatus: { type: 'string', enum: ['COMPLETED', 'IN_PROGRESS', 'PENDING', 'NOT_STARTED'], description: 'Filter phases by status. PENDING = anything not completed.' },
            milestoneName: { type: 'string', description: 'Filter by milestone name (substring), e.g. "Payment", "Handover", "Mobilization", "Site Kick-Off".' },
            milestoneStatus: { type: 'string', enum: ['COMPLETED', 'IN_PROGRESS', 'PENDING', 'NOT_STARTED'], description: 'Filter milestones by status. Use PENDING for "pending payment"/"pending action" (= not completed).' },
            hasCompleted: { type: 'string', description: 'Project has a COMPLETED milestone whose name contains this, e.g. "Mobilization Advance".' },
            missing: { type: 'string', description: 'Project does NOT have completed a milestone whose name contains this, e.g. "Site Kick-Off". Combine with hasCompleted for "received X but not yet Y".' },
            overdue: { type: 'boolean', description: 'Only milestones past their due date (endDate < today) and not completed. Use for "overdue" / "behind schedule".' },
            dueWithinDays: { type: 'number', description: 'Only milestones due within the next N days (endDate between today and today+N), not yet completed. Use for "due in the next 2 weeks" (N=14).' },
            dueBefore: { type: 'string', description: 'ISO date (YYYY-MM-DD). Only milestones due on/before this date.' },
            dueAfter: { type: 'string', description: 'ISO date (YYYY-MM-DD). Only milestones due on/after this date.' },
            region: { type: 'string', description: 'Filter matched projects by normalized region: North, South, East, West.' },
            zone: { type: 'string', description: 'Filter matched projects by raw zone/hub, e.g. "Gurgaon".' },
            city: { type: 'string', description: 'Filter matched projects by city, e.g. "Gurugram".' },
            owner: { type: 'string', description: 'Filter matched projects by owner/team.' },
            groupBy: { type: 'string', enum: ['region', 'city', 'owner', 'currentPhase'], description: 'Group the matching projects and return counts per group. Use "region" for any "zone-wise ..." question (never raw zone), "currentPhase" for a stage funnel.' },
          },
        },
      },
    },
  ];

  private buildSystemInstruction(customerId: string): string {
    return `${SYSTEM_PROMPT}

${TABLE_FORMAT_INSTRUCTION}

You are the PLANNER. Decide which tool(s) to call to answer the user, call them, read the results, then write the final answer.
Rules:
- The customerId is "${customerId}" and is already known — never ask the user for it.
- For totals, counts, averages, rankings, filtered sums/lists, or ANY chart/comparison request, you MUST call projectAnalytics. Do not calculate numbers yourself.
- For "leads" vs "projects" use projectAnalytics type="lead"/"project"; for "active" use active=true; for "unassigned"/"no owner" use ownerMissing=true.
- For date windows ("last 30 days", "last 3/6 months", "this quarter") use lastNDays/lastNMonths (or createdAfter/createdBefore) on projectAnalytics; for trends over time add bucketBy="month" or "quarter" and present the timeline as a table.
- NEVER compute averages/totals yourself — read the exact numbers from the tool result. For a timeline bucket use its "formattedAverage" / "formattedMedian" verbatim. When a bucket's "outliers" is > 0, the mean is skewed by bad data-entry records, so present the "formattedMedian" as the typical value and add a short note that the average is skewed by outlier records.
- For COUNT or group-by questions, the answer table must show the group and its COUNT. Do NOT add an area or value column unless the user explicitly asked about area or value.
- For a two-dimension "X-wise breakdown BY Y" / "X by Y" question, set BOTH groupBy=X and groupBy2=Y and render the returned "pivot" as a table (X = rows, Y = columns). NEVER invent a single row/label to fake a breakdown — if you only grouped one field, present only that field.
- "Started" = created in the period (projectAnalytics, lastNMonths). Any "<stage> completed/done/reached" = a COMPLETED milestone of that name: call queryProjectFlow with milestoneName=<the stage>, milestoneStatus="COMPLETED", groupBy="region" (add dueAfter/dueBefore for a time window).
- BUSINESS DEFINITIONS:
  * "delayed" / "behind schedule" / "overdue" / "not completed on time" = a milestone whose due date (endDate) has passed and is not COMPLETED → use queryProjectFlow overdue=true.
  * "pending" (payment/approval/action) = that milestone not COMPLETED → milestoneStatus="PENDING".
  * "project completed" / "completion" / "completion rate" = the "Project Closure" milestone is COMPLETED → queryProjectFlow milestoneName="Project Closure", milestoneStatus="COMPLETED". A completion RATE = completed projects ÷ total projects (call once for completed, once for total).
  * "GM" = the team member whose role is General Manager. For "per GM / GM workload / which GM" use projectAnalytics groupByRole="General Manager".
- PERSON queries (a name like "Nitish"): "projects of <Name>" / "<Name>'s projects" → projectAnalytics teamMember="<Name>" (default to team member). "where <Name> is a team member" → teamMember="<Name>". "where <Name> is the customer" → customerName="<Name>". "where <Name> is the GM" → teamMember="<Name>", role="General Manager". A bare name is ambiguous, so answer the team-member reading and you may note they can also check customer/GM.
- COMPARISONS ("A vs B", "A compared to B", "A versus B"): make ONE tool call per side, then present the results as columns side by side in a single table. Never decline a comparison just because it has two parts — answer each part with its own tool call.
- The real project-flow vocabulary (map the user's words to the CLOSEST of these before calling queryProjectFlow; use a distinctive substring):
  Phases: Pre-Sales, Design-Sales, Design Delivery, Execution, Handover.
  Milestones: Lead Creation, Customer Info Received, Client Brief Meeting, Commercial Proposal, Commercial Approval, Design Fee-Advance, Initial Design Kit, Final Design Kit, BOQ Approval, Firm BOQ Approval, Procurement Start, Site Mobilisation Advance, Site Kick-Off, GFC-Set 1, GFC-Set 2, Milestone -1/-2/-3 Payment Due, Snags Rectification, Sign off, Handover, Project Closure, Final Bill.
  Example: "initial design kit completed" -> milestoneName="Initial Design Kit", milestoneStatus="COMPLETED".
- When the user names one or more specific project codes (e.g. "project code 2022072479"), call projectAnalytics with projectCodes set to those codes — never use searchProjects for an exact code.
- FOLLOW-UPS about "the above / these / those / them" projects refer to the projects in the MOST RECENT list in this conversation. Read every project code from that previous answer and call projectAnalytics with projectCodes set to ALL of them, then answer for that exact set (e.g. show code + owner). Do not say "no projects match" — the codes are in the conversation above.
- DRILL-DOWN: if the user asks to "list those / these / the projects" right after a COUNT or grouped/aggregate answer (e.g. "which region has the most overdue milestones" → then "list those projects"), RE-RUN the same tool that produced that count, but as a LIST, applying the SAME filters from that earlier turn. Example: after "most overdue milestones = Unassigned", answer "list those" by calling queryProjectFlow with operation="listProjects", overdue=true, region="Unassigned". Never reply that you lack the details — you can always re-query with the filters already established in the conversation.
- For descriptive questions about a specific project, call searchProjects.
- When listing projects returned by queryProjectFlow, render the table with ONLY these columns: | Project Code | Project Name | Current Phase |. Do NOT add a "Matched Milestones" or any milestone column — it is noise in a list; the user opens a project's Timeline for milestone detail.
- For questions about project WORKFLOW/STAGE progress — milestone status, pending payments, overdue or upcoming-due milestones, "behind schedule", which projects are at/stuck-at a stage, mobilization/handover milestones, or the stage funnel — call queryProjectFlow. A milestone's endDate is its due date: use overdue=true for "overdue"/"behind schedule" and dueWithinDays for "due in the next N days".
- Base every fact and number ONLY on tool results. If the tools return nothing relevant, say you don't have that information.
- When you have enough information, reply with the final answer as a markdown table (if multiple records) or plain text (if single fact).`;
  }

  // Run the tool-calling loop until the model has everything it needs to answer.
  // Returns { messages, trace, done }: `done` is the final answer text if the model
  // produced one without needing a final streamed turn, else null. `messages` holds
  // the full conversation (incl. tool results) ready for a final answer turn.
  private async resolveTools(
    messages: any[],
    customerId: string,
  ): Promise<{ trace: PlannerResult['trace']; finalText: string | null; steps: number }> {
    const trace: PlannerResult['trace'] = [];

    for (let step = 1; step <= MAX_STEPS; step++) {
      const assistant = await this.openaiService.openRouterToolTurn(messages, this.tools);
      const toolCalls = assistant?.tool_calls;

      if (!toolCalls || toolCalls.length === 0) {
        return { trace, finalText: assistant?.content ?? '', steps: step };
      }

      messages.push(assistant);

      for (const call of toolCalls) {
        let args: any = {};
        try {
          args = JSON.parse(call.function?.arguments || '{}');
        } catch {
          args = {};
        }
        const result = await this.executeTool(call.function?.name, args, customerId);
        trace.push({ tool: call.function?.name, args, result });
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
      }
    }

    return { trace, finalText: null, steps: MAX_STEPS };
  }

  async run(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
    referencedCodes?: string[];
    userName?: string;
  }): Promise<PlannerResult> {
    const { question, customerId, history = [], referencedCodes = [], userName } = params;

    const messages: any[] = [
      { role: 'system', content: this.buildSystemInstruction(customerId) },
      ...history,
    ];
    // Resolve first-person queries to the logged-in user.
    if (userName) {
      messages.push({
        role: 'system',
        content: `The current user is "${userName}". Interpret "my", "me", "mine", "I" as this person. For "my projects" / "projects assigned to me", call projectAnalytics with teamMember="${userName}".`,
      });
    }
    // Hand the LLM the exact codes for an "above/these" follow-up so it only has to
    // pass them through (not scrape them out of a previous list).
    if (referencedCodes.length) {
      messages.push({
        role: 'system',
        content: `The user is referring to these specific projects from earlier in the conversation. Use projectAnalytics with projectCodes set to EXACTLY these, then apply any sorting/metric/topN the user asked for: ${JSON.stringify(referencedCodes)}`,
      });
    }
    messages.push({ role: 'user', content: question });

    const { trace, finalText, steps } = await this.resolveTools(messages, customerId);
    return {
      answer: finalText ?? 'I could not complete the request within the allowed number of steps.',
      trace,
      steps,
    };
  }

  // Streaming variant: resolves tools the same way, then STREAMS the final answer
  // token-by-token to the HTTP response (for the chat UI's typing effect).
  async runStream(params: {
    question: string;
    customerId: string;
    history?: Array<{ role: string; content: string }>;
    res: Response;
    onAnswer?: (answer: string) => void;
  }): Promise<void> {
    const { question, customerId, history = [], res, onAnswer } = params;

    const messages: any[] = [
      { role: 'system', content: this.buildSystemInstruction(customerId) },
      ...history,
      { role: 'user', content: question },
    ];

    // Drive the tool loop. Its final turn already produces the answer text, so we
    // reuse that instead of re-generating it with a second streaming LLM call —
    // saving one full generation round-trip per request.
    const { finalText } = await this.resolveTools(messages, customerId);
    const answer = finalText || 'I could not complete the request within the allowed number of steps.';
    onAnswer?.(answer);

    // Emulate token streaming by writing the answer in small chunks so the UI keeps
    // its typing effect, without the cost of a real second streamed generation.
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.flushHeaders?.();

    const tokens = answer.split(/(\s+)/); // keep whitespace tokens
    for (let i = 0; i < tokens.length; i += 3) {
      if (res.writableEnded) break;
      res.write(tokens.slice(i, i + 3).join(''));
      (res as any).flush?.();
      await new Promise((r) => setTimeout(r, 6));
    }
    if (!res.writableEnded) res.end();
  }

  private async executeTool(name: string, args: any, customerId: string) {
    try {
      if (name === 'searchProjects') {
        const embedding = await this.openaiService.generateEmbedding(args.query || '');
        const hits = await this.qdrantService.search(COLLECTION, {
          vector: embedding,
          limit: 50,
          filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
        });
        const rawChunks = hits
          .sort((a: any, b: any) => b.score - a.score)
          .map((h: any) => h.payload?.text as string);
        const reranked = await this.rerankService.rerank(args.query || '', rawChunks, 10);
        return reranked.slice(0, 10);
      }

      if (name === 'projectAnalytics') {
        return this.projectAnalyticsService.analyze({
          customerId,
          metric: args.metric,
          topN: args.topN,
          projectCodes: args.projectCodes,
          teamMember: args.teamMember,
          role: args.role,
          groupBy: args.groupBy,
          groupBy2: args.groupBy2,
          groupByRole: args.groupByRole,
          city: args.city,
          state: args.state,
          zone: args.zone,
          region: args.region,
          stage: args.stage,
          subStage: args.subStage,
          owner: args.owner,
          channel: args.channel,
          projectStatus: args.projectStatus,
          companyName: args.companyName,
          customerName: args.customerName,
          minValue: args.minValue,
          maxValue: args.maxValue,
          equalsValue: args.equalsValue,
          minArea: args.minArea,
          maxArea: args.maxArea,
          equalsArea: args.equalsArea,
          type: args.type,
          active: args.active,
          ownerMissing: args.ownerMissing,
          dateField: args.dateField,
          lastNDays: args.lastNDays,
          lastNMonths: args.lastNMonths,
          createdAfter: args.createdAfter,
          createdBefore: args.createdBefore,
          bucketBy: args.bucketBy,
        });
      }

      if (name === 'queryProjectFlow') {
        return this.projectFlowService.analyze({
          customerId,
          operation: args.operation,
          projectCode: args.projectCode,
          phaseName: args.phaseName,
          phaseStatus: args.phaseStatus,
          milestoneName: args.milestoneName,
          milestoneStatus: args.milestoneStatus,
          hasCompleted: args.hasCompleted,
          missing: args.missing,
          overdue: args.overdue,
          dueWithinDays: args.dueWithinDays,
          dueBefore: args.dueBefore,
          dueAfter: args.dueAfter,
          region: args.region,
          zone: args.zone,
          city: args.city,
          owner: args.owner,
          groupBy: args.groupBy,
        });
      }

      return { error: `Unknown tool: ${name}` };
    } catch (err: any) {
      return { error: err?.message || 'tool execution failed' };
    }
  }
}
