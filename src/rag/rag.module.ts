import { Module } from '@nestjs/common';
import { ChatModule } from '../chat/chat.module';
import { RagController } from './rag.controller';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { ClaudeService } from './claude.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { ProjectFlowService } from './project-flow.service';
import { PlannerService } from './planner.service';
import { RerankService } from './rerank.service';
import { ProjectQueryService } from './project-query.service';
import { ProjectAgentService } from './project-agent.service';
import { AgentGraphService } from './agent-graph.service';
import { VoiceService } from './voice.service';
import { IngestionController } from './ingest/ingestion.controller';
import { IngestionService } from './ingest/ingestion.service';
import { PgService } from './ingest/pg.service';
import { WatermarkStore } from './ingest/watermark.store';
import { SemanticCacheService } from './semantic-cache.service';
import { ApiUserModule } from 'src/api/api.module';
import { WbGuard } from 'src/core/guards/wb-guard.guard';

@Module({
  imports: [ChatModule, ApiUserModule],
  controllers: [RagController, IngestionController],
  providers: [
    QdrantService,
    OpenaiService,
    SemanticCacheService,
    ClaudeService,
    GuardrailService,
    ProjectAnalyticsService,
    ProjectFlowService,
    PlannerService,
    RerankService,
    ProjectQueryService,
    ProjectAgentService,
    AgentGraphService,
    VoiceService,
    IngestionService,
    PgService,
    WatermarkStore,
    WbGuard,
  ],
})
export class RagModule {}
