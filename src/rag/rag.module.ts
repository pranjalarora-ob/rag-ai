import { Module } from '@nestjs/common';
import { RagController } from './rag.controller';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { ClaudeService } from './claude.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { PlannerService } from './planner.service';
import { RerankService } from './rerank.service';
import { ProjectQueryService } from './project-query.service';
import { ProjectAgentService } from './project-agent.service';
import { IngestionController } from './ingest/ingestion.controller';
import { IngestionService } from './ingest/ingestion.service';
import { PgService } from './ingest/pg.service';
import { WatermarkStore } from './ingest/watermark.store';
import { SemanticCacheService } from './semantic-cache.service';

@Module({
  controllers: [RagController, IngestionController],
  providers: [
    QdrantService,
    OpenaiService,
    SemanticCacheService,
    ClaudeService,
    GuardrailService,
    ProjectAnalyticsService,
    PlannerService,
    RerankService,
    ProjectQueryService,
    ProjectAgentService,
    IngestionService,
    PgService,
    WatermarkStore,
  ],
})
export class RagModule {}
