import { Module } from '@nestjs/common';
import { RagController } from './rag.controller';
import { QdrantService } from './qdrant.service';
import { OpenaiService } from './openai.service';
import { GuardrailService } from './guardrail.service';
import { ProjectAnalyticsService } from './project-analytics.service';
import { PlannerService } from './planner.service';

@Module({
  controllers: [RagController],
  providers: [QdrantService, OpenaiService, GuardrailService, ProjectAnalyticsService, PlannerService],
})
export class RagModule {}
