import { Controller, Param, Post, Query } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { IngestionService } from './ingestion.service';

@ApiTags('RAG Ingestion')
@Controller('rag/ingest')
export class IngestionController {
  constructor(private readonly ingestion: IngestionService) { }

  @ApiOperation({ summary: 'Backfill last N months (default 6) for a source: project | boq | project-flow-detail' })
  @Post('backfill/:source')
  backfill(@Param('source') source: string, @Query('months') months?: string) {
    return this.ingestion.backfill(source, months ? Number(months) : 6);
  }

  @ApiOperation({ summary: 'Incremental sync — only rows changed since the last run. Cron this every 1-5 min.' })
  @Post('sync/:source')
  sync(@Param('source') source: string) {
    return this.ingestion.sync(source);
  }

  @ApiOperation({ summary: 'Real-time: (re)ingest one entity by id. Call from a create/update webhook.' })
  @Post('entity/:source/:id')
  entity(@Param('source') source: string, @Param('id') id: string) {
    return this.ingestion.ingestEntity(source, id);
  }
}
