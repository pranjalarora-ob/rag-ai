import { Module } from '@nestjs/common';
import { RagController } from './rag.controller';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { ClaudeService } from './claude.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { PlannerService } from './planner.service';
import { RerankService } from './rerank.service';
import { IngestionController } from './ingest/ingestion.controller';
import { IngestionService } from './ingest/ingestion.service';
import { PgService } from './ingest/pg.service';
import { WatermarkStore } from './ingest/watermark.store';

@Module({
  controllers: [RagController, IngestionController],
  providers: [
    QdrantService,
    OpenaiService,
    ClaudeService,
    GuardrailService,
    ProjectAnalyticsService,
    PlannerService,
    RerankService,
    IngestionService,
    PgService,
    WatermarkStore,
  ],
})
export class RagModule {}
