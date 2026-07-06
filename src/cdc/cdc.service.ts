import { Injectable, Logger } from '@nestjs/common';
import { DebeziumEvent } from './interfaces/debezium-event.interface';

@Injectable()
export class CdcService {
  private readonly logger = new Logger(CdcService.name);

  async process(topic: string, event: DebeziumEvent<any>) {
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

  private async handleCreate(topic: string, data: any) {
    this.logger.log(`CREATE on ${topic}`);
    this.logger.debug(data);

    // TODO:
    // Update Search
    // Update Qdrant
    // Send Notification
  }

  private async handleUpdate(
    topic: string,
    before: any,
    after: any,
  ) {
    this.logger.log(`UPDATE on ${topic}`);

    this.logger.debug({
      before,
      after,
    });

    // TODO
  }

  private async handleDelete(
    topic: string,
    before: any,
  ) {
    this.logger.log(`DELETE on ${topic}`);

    this.logger.debug(before);

    // TODO
  }

  private async handleSnapshot(
    topic: string,
    after: any,
  ) {
    this.logger.log(`SNAPSHOT ${topic}`);

    this.logger.debug(after);
  }
}