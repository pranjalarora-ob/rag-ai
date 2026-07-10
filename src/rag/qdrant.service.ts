import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { QdrantClient } from '@qdrant/js-client-rest';
import axios from 'axios';
import { v5 as uuidV5 } from 'uuid';
import { AddSchemaIndexDto, Point, SearchQdrantDto } from './dto/qdrant.dto';

@Injectable()
export class QdrantService {
  private QDRANT_URL: string;
  private QDRANT_API_KEY: string;
  private client: QdrantClient;
  EMBEDDING_DIM = 1536; // openai/text-embedding-3-small

  constructor(private readonly configService: ConfigService) {
    this.QDRANT_URL = this.configService.get('QDRANT_URL') || 'http://localhost:6333';
    this.QDRANT_API_KEY = this.configService.get('QDRANT_API_KEY') || '';
    this.client = new QdrantClient({ url: this.QDRANT_URL, apiKey: this.QDRANT_API_KEY, checkCompatibility: false });
  }

  generateChunkId(originalId: string, chunkIndex: number) {
    const NAMESPACE = 'a7e6482f-6c26-4cf6-bde3-7151fc4a0a94';
    return uuidV5(`${originalId}_${chunkIndex}`, NAMESPACE);
  }

  async createCollection(collection: string) {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some((c) => c.name === collection);
    if (!exists) {
      return this.client.createCollection(collection, {
        vectors: { size: this.EMBEDDING_DIM, distance: 'Cosine' },
      });
    }
    return false;
  }

  // Sentence-aware chunking with token overlap.
  chunkTextv2(text: string, maxTokens = 700, overlapTokens = 100) {
    const sentences = text.split(/(?<=[.?!])\s+/);
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentTokenCount = 0;
    const estimateTokens = (str: string) => Math.ceil(str.split(/\s+/).length * 1.3);

    for (const sentence of sentences) {
      const sentenceTokens = estimateTokens(sentence);
      if (currentTokenCount + sentenceTokens > maxTokens) {
        chunks.push(currentChunk.join(' '));
        let overlap: string[] = [];
        let overlapCount = 0;
        for (let i = currentChunk.length - 1; i >= 0; i--) {
          const tokens = estimateTokens(currentChunk[i]);
          if (overlapCount + tokens > overlapTokens) break;
          overlap.unshift(currentChunk[i]);
          overlapCount += tokens;
        }
        currentChunk = [...overlap];
        currentTokenCount = overlapCount;
      }
      currentChunk.push(sentence);
      currentTokenCount += sentenceTokens;
    }
    if (currentChunk.length) chunks.push(currentChunk.join(' '));
    return chunks;
  }

  async upsert(collection: string, points: Point[]) {
    await this.createCollection(collection);
    try {
      return await this.client.upsert(collection, { points });
    } catch (error) {
      console.error('Qdrant upsert error:', error);
      throw error;
    }
  }

  async search(collection: string, options: SearchQdrantDto) {
    try {
      return await this.client.search(collection, options);
    } catch (error: any) {
      // A collection that doesn't exist yet returns 404 — treat as no results.
      if (error?.status === 404 || /not found|doesn't exist/i.test(error?.message || '')) {
        return [];
      }
      throw error;
    }
  }

  // Fetch ALL points matching a filter (paginates). Used by analytics, which needs every record.
  async scrollAll(collection: string, filter?: any, pageSize = 250) {
    const points: any[] = [];
    let offset: any = undefined;
    try {
      do {
        const res = await this.client.scroll(collection, {
          filter,
          limit: pageSize,
          offset,
          with_payload: true,
          with_vector: false,
        });
        points.push(...res.points);
        offset = res.next_page_offset;
      } while (offset !== null && offset !== undefined);
    } catch (error: any) {
      console.error('Qdrant scrollAll error details:', {
        status: error?.status,
        message: error?.message,
        data: error?.data,
      });
      if (error?.status === 404) return [];
      throw error;
    }
    return points;
  }

  async addIndex(collection: string, addSchemaIndexDto: AddSchemaIndexDto) {
    try {
      console.log('[QdrantService] addIndex call:', { collection, addSchemaIndexDto });
      const res = await axios.put(
        `${this.QDRANT_URL}/collections/${collection}/index?wait=true`,
        {
          field_name: addSchemaIndexDto.field,
          field_schema: addSchemaIndexDto.schema,
        },
        { headers: this.QDRANT_API_KEY ? { 'api-key': this.QDRANT_API_KEY } : {} },
      );
      return res.data;
    } catch (error: any) {
      console.error('Qdrant addIndex error details:', {
        status: error?.response?.status || error?.status,
        message: error?.response?.statusText || error?.message,
        data: error?.response?.data || error?.data,
      });
      throw error;
    }
  }

  async clearCollection(collection: string) {
    const res = await axios.post(
      `${this.QDRANT_URL}/collections/${collection}/points/delete?wait=true`,
      { filter: {} },
      { headers: this.QDRANT_API_KEY ? { 'api-key': this.QDRANT_API_KEY } : {} },
    );
    return res.data;
  }
}
