import {
  Controller, Post, Body, Param, Get, Res, Patch,
  BadRequestException, InternalServerErrorException,
} from '@nestjs/common';
import { ApiOperation, ApiProduces, ApiResponse, ApiTags } from '@nestjs/swagger';
import { Response } from 'express';
import { ChatDto, IngestVectorDataDto, PlannerDto } from './dto/rag.dto';
import { AddSchemaIndexDto, Point } from './dto/qdrant.dto';
import { ChatCompletionRequestMessage } from './dto/openai.dto';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService, AnalyticsQuery } from './project-analytics.service';
import { PlannerService } from './planner.service';
import { SYSTEM_PROMPT, COLLECTION } from './constants';

@ApiTags('RAG')
@Controller('rag')
export class RagController {
  constructor(
    private readonly qdrantService: QdrantService,
    private readonly openaiService: OpenaiService,
    private readonly guardrailService: GuardrailService,
    private readonly projectAnalyticsService: ProjectAnalyticsService,
    private readonly plannerService: PlannerService,
  ) {}

  // ============ Ingestion ============

  @ApiOperation({ summary: 'Ingest documents into the Qdrant vector DB (chunk + embed)' })
  @Post('openai/qdrant/ingest')
  async ingest(@Body() dto: IngestVectorDataDto) {
    if (!Array.isArray(dto.documents)) {
      throw new BadRequestException("Invalid 'documents' format");
    }

    const points: Point[] = [];
    for (const doc of dto.documents) {
      const chunks = this.qdrantService.chunkTextv2(doc.text);
      for (let i = 0; i < chunks.length; i++) {
        const embedding = await this.openaiService.generateGeminiEmbedding(chunks[i]);
        points.push({
          id: this.qdrantService.generateChunkId(doc.id, i),
          vector: embedding,
          payload: {
            ...doc.metadata,
            chunk_index: i,
            customerId: dto.customerId,
            total_chunks: chunks.length,
            text: chunks[i],
            original_id: doc.id,
          },
        });
      }
    }

    await this.qdrantService.upsert(dto.collection, points);
    return { message: 'Ingestion successful', count: points.length };
  }

  @ApiOperation({ summary: 'Clear a Qdrant collection' })
  @Get('qdrant/:collection/clear')
  clearCollection(@Param('collection') collection: string) {
    return this.qdrantService.clearCollection(collection);
  }

  @ApiOperation({ summary: 'Add a payload index (required before filtering on a field, e.g. customerId)' })
  @Patch('qdrant/:collection/add-index')
  addIndex(@Param('collection') collection: string, @Body() body: AddSchemaIndexDto) {
    return this.qdrantService.addIndex(collection, body);
  }

  // ============ Tools ============

  @ApiOperation({ summary: 'Analytics tool — exact totals, top-N, averages, filtered sums (no LLM math)' })
  @Post('projects/analytics')
  analytics(@Body() query: AnalyticsQuery) {
    if (!query?.customerId) throw new BadRequestException('customerId is required');
    return this.projectAnalyticsService.analyze(query);
  }

  // ============ Planner (the thinker) ============

  @ApiOperation({ summary: 'Planner agent — picks tools (RAG / analytics), loops, drafts answer. Returns answer + trace.' })
  @Post('planner')
  planner(@Body() body: PlannerDto) {
    return this.plannerService.run({ question: body.question, customerId: body.customerId });
  }

  // ============ Chat (streaming, with routing) ============

  @ApiOperation({ summary: 'Chat — routes aggregate questions to the analytics tool, lookups to RAG. Streams text.' })
  @ApiProduces('text/event-stream')
  @ApiResponse({ status: 200, description: 'Streamed plain-text answer' })
  @Post('openai/chat')
  async chat(@Body() chatDto: ChatDto, @Res() res: Response) {
    const { question, customerId } = chatDto;

    try {
      if (this.guardrailService.isPolicyViolation(question)) {
        throw new BadRequestException('Query violates company policy.');
      }

      const route = await this.classifyQuery(question);

      // ---- Analytics route: exact numbers from the tool, phrased by the LLM ----
      if (route.type === 'analytics') {
        const analytics = await this.projectAnalyticsService.analyze({
          customerId,
          metric: route.metric,
          topN: route.topN,
          teamMember: route.teamMember,
          role: route.role,
          groupBy: route.groupBy,
        });

        const messages: ChatCompletionRequestMessage[] = [
          {
            role: 'system',
            content: `${SYSTEM_PROMPT}\n\nAnswer using these EXACT pre-computed results. Do not recompute or invent numbers. Format large numbers with commas.\n\nResults (JSON):\n${JSON.stringify(analytics)}`,
          },
          { role: 'user', content: question },
        ];
        await this.openaiService.streamOpenRouter(messages, res);
        return;
      }

      // ---- RAG route: retrieve relevant projects, answer from them ----
      const embedding = await this.openaiService.generateGeminiEmbedding(question.trim());
      const hits = await this.qdrantService.search(COLLECTION, {
        vector: embedding,
        limit: 100,
        filter: { must: [{ key: 'customerId', match: { value: customerId } }] },
      });
      const rawChunks = hits.sort((a: any, b: any) => b.score - a.score).map((r: any) => r.payload?.text as string);
      const context = await this.openaiService.truncateByTokens(rawChunks, 12000);

      const messages: ChatCompletionRequestMessage[] = [
        {
          role: 'system',
          content: `${SYSTEM_PROMPT}\n\nUse ONLY the context below to answer. If it asks for numbers, return exact values from the context; do not invent.\n\nContext:\n${context}`,
        },
        { role: 'user', content: question },
      ];
      await this.openaiService.streamOpenRouter(messages, res);
    } catch (err) {
      console.error(err);
      if (!res.headersSent) {
        throw new InternalServerErrorException('Chat service failed');
      }
      if (!res.writableEnded) res.end();
    }
  }

  // ============ Helpers ============

  // Decide analytics vs RAG and extract analytics params. Keyword fast-path avoids an
  // extra LLM call for obvious cases; otherwise an LLM classifier extracts name filters.
  private async classifyQuery(question: string): Promise<{
    type: 'analytics' | 'rag';
    metric?: 'estimatedValue' | 'boqValue';
    topN?: number;
    teamMember?: string;
    role?: string;
    groupBy?: 'stage' | 'city' | 'owner';
  }> {
    const q = question.toLowerCase();
    const aggregateHint = /\b(total|sum|average|avg|count|how many|highest|lowest|top \d+|top\s|rank|most|least|combined)\b/.test(q);
    const lookupHint = /\b(who|what stage|which stage|describe|details? of|tell me about|status of)\b/.test(q);
    const hasNameFilter = /\b(where|whose|by)\b/.test(q) && /\b(manager|lead|member|owner|head)\b/.test(q);

    if (aggregateHint && !lookupHint && !hasNameFilter) {
      const topMatch = q.match(/top\s+(\d+)/);
      const metric: 'estimatedValue' | 'boqValue' = /\bboq\b/.test(q) ? 'boqValue' : 'estimatedValue';
      let groupBy: 'stage' | 'city' | 'owner' | undefined;
      if (/\bper city|by city|each city\b/.test(q)) groupBy = 'city';
      else if (/\bper stage|by stage|each stage\b/.test(q)) groupBy = 'stage';
      else if (/\bper owner|by owner|each owner|by team\b/.test(q)) groupBy = 'owner';
      return { type: 'analytics', metric, topN: topMatch ? Number(topMatch[1]) : undefined, groupBy };
    }

    const prompt = `You are a router for a project assistant. Classify the user's question and extract parameters.
Return STRICT JSON only — no markdown.

Schema:
{ "type": "analytics" | "rag", "metric": "estimatedValue" | "boqValue", "topN": number, "teamMember": string, "role": string, "groupBy": "stage" | "city" | "owner" }

Use "analytics" for totals, sums, averages, counts, rankings, or filtered aggregates ACROSS many projects.
Use "rag" for descriptive lookups about a specific project.

Examples:
"total cost of all projects" -> {"type":"analytics","metric":"estimatedValue"}
"top 5 projects by value" -> {"type":"analytics","metric":"estimatedValue","topN":5}
"total estimated value where Ashish is the design manager" -> {"type":"analytics","metric":"estimatedValue","teamMember":"Ashish","role":"Design Manager"}
"who is the design manager on yatra 87" -> {"type":"rag"}

Question: "${question}"`;

    try {
      const raw = await this.openaiService.openRouterGenerate(prompt);
      const json = raw.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(json);
      if (parsed?.type === 'analytics' || parsed?.type === 'rag') return parsed;
    } catch (err) {
      console.error('classifyQuery failed, defaulting to rag:', err);
    }
    return { type: 'rag' };
  }
}
