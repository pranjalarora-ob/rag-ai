import { Injectable, Logger } from '@nestjs/common';
import { DebeziumEvent } from './interfaces/debezium-event.interface';
import { IngestionService } from '../rag/ingest/ingestion.service';
import { OpenaiService } from '../rag/openai.service';
import { QdrantService } from '../rag/qdrant.service';
import { COLLECTION } from '../rag/constants';

type CdcSource = 'project' | 'boq';

@Injectable()
export class CdcService {
  private readonly logger = new Logger(CdcService.name);

  constructor(
    private readonly ingestionService: IngestionService,
    private readonly openaiService: OpenaiService,
    private readonly qdrantService: QdrantService,
  ) {}

  async process(topic: string, event: DebeziumEvent<any>) {
    if (!event || !event.payload) {
      this.logger.warn(`Malformed CDC event on topic ${topic}`);
      return;
    }
    const payload = event.payload;

    switch (payload.op) {
      case 'c':
        await this.handleCreate(topic, payload.after);
        break;

      case 'u':
        await this.handleUpdate(topic, payload.before, payload.after);
        break;

      case 'd':
        await this.handleDelete(topic, payload.before);
        break;

      case 'r':
        await this.handleSnapshot(topic, payload.after);
        break;
    }
  }

  private topicToSource(topic: string): CdcSource | null {
    if (topic.includes('projects')) return 'project';
    if (topic.includes('quotations')) return 'boq';
    return null;
  }

  private entityId(entity: any): string | null {
    if (!entity) return null;
    if (typeof entity.id === 'string') return entity.id;
    if (typeof entity.id === 'number') return String(entity.id);
    return null;
  }

  private extractText(data: any): string {
    if (!data) return '';
    if (typeof data === 'string') return data.trim();
    if (typeof data.text === 'string' && data.text.trim()) return data.text.trim();
    const chunks: string[] = [];
    const build = (label: string, value: any) => {
      if (value === undefined || value === null) return;
      const cleaned = String(value).trim();
      if (!cleaned) return;
      chunks.push(`${label}: ${cleaned}`);
    };

    build('id', data.id);
    build('name', data.name || data.project_name || data.company_name || data.code);
    build('type', data.type);
    build('status', data.status || data.project_status);
    build('stage', data.stage);
    build('owner', data.owner);
    build('city', data.city);
    build('description', data.desc || data.description || data.notes);
    build('project', data.project_name || data.project_company);
    build('code', data.code || data.boq_code);

    if (data.customer_info) {
      const customer = typeof data.customer_info === 'string' ? data.customer_info : JSON.stringify(data.customer_info);
      build('customer_info', customer);
    }

    if (chunks.length) return chunks.join('\n');
    return JSON.stringify(data, null, 2);
  }

  private async upsertTextVector(source: CdcSource, id: string, text: string, data: any) {
    const vector = await this.openaiService.generateEmbedding(text);
    const point = {
      id,
      vector,
      payload: {
        docType: source,
        original_id: id,
        text,
        ...this.buildPayloadMetadata(source, data),
      },
    };

    await this.qdrantService.upsert(COLLECTION, [point]);
  }

  private buildPayloadMetadata(source: CdcSource, data: any) {
    const metadata: Record<string, any> = {};
    if (!data || typeof data !== 'object') return metadata;
    if (source === 'project') {
      metadata.projectId = this.entityId(data);
      metadata.projectCode = data.code;
      metadata.projectName = data.name || data.project_name;
      metadata.companyName = data.company_name;
      metadata.customerId = data.account_id;
    } else {
      metadata.boqId = this.entityId(data);
      metadata.projectId = data.project_id;
      metadata.boqCode = data.code;
      metadata.boqType = data.type;
      metadata.status = data.status;
      metadata.customerId = data.account_id;
    }
    return metadata;
  }

  private async handleCreate(topic: string, data: any) {
    const source = this.topicToSource(topic);
    const id = this.entityId(data);
    const text = this.extractText(data);

    this.logger.log(`CDC CREATE event on ${topic}`);
    this.logger.debug({ data });

    if (!source || !id) {
      this.logger.warn(`Skipping CDC CREATE for topic ${topic}: unsupported topic or missing id`);
      return;
    }

    if (text) {
      await this.upsertTextVector(source, id, text, data);
      return;
    }

    await this.ingestionService.ingestEntity(source, id);
  }

  private async handleUpdate(
    topic: string,
    before: any,
    after: any,
  ) {
    const source = this.topicToSource(topic);
    const entity = after || before;
    const id = this.entityId(entity);
    const text = this.extractText(entity);

    this.logger.log(`CDC UPDATE event on ${topic}`);
    this.logger.debug({
      before,
      after,
    });

    if (!source || !id) {
      this.logger.warn(`Skipping CDC UPDATE for topic ${topic}: unsupported topic or missing id`);
      return;
    }

    if (text) {
      await this.upsertTextVector(source, id, text, entity);
      return;
    }

    await this.ingestionService.ingestEntity(source, id);
  }

  private async handleDelete(
    topic: string,
    before: any,
  ) {
    const source = this.topicToSource(topic);
    const id = this.entityId(before);

    this.logger.log(`CDC DELETE event on ${topic}`);
    this.logger.debug({ before });

    if (!source || !id) {
      this.logger.warn(`Skipping CDC DELETE for topic ${topic}: unsupported topic or missing id`);
      return;
    }

    this.logger.log(`Deleting stale CDC vector for ${topic} id=${id}`);
    try {
      await this.qdrantService.deletePoint(COLLECTION, id);
      this.logger.log(`Deleted stale point ${id} from Qdrant collection ${COLLECTION}`);
    } catch (error) {
      this.logger.error(
        `Failed to delete stale point ${id} from Qdrant collection ${COLLECTION}: ${error?.message || error}`,
      );
    }
  }

  private async handleSnapshot(
    topic: string,
    after: any,
  ) {
    this.logger.log(`SNAPSHOT ${topic}`);

    this.logger.debug(after);
  }
}