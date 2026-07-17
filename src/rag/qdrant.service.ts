import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { v5 as uuidV5 } from 'uuid';
import { AddSchemaIndexDto, Point, SearchQdrantDto } from './dto/qdrant.dto';

@Injectable()
export class QdrantService {
  private QDRANT_URL: string;
  private QDRANT_API_KEY: string;
  EMBEDDING_DIM = 1536; // openai/text-embedding-3-small

  constructor(private readonly configService: ConfigService) {
    this.QDRANT_URL = (this.configService.get('QDRANT_URL') || 'http://localhost:6333').replace(/\/+$/, '');
    this.QDRANT_API_KEY = this.configService.get('QDRANT_API_KEY') || '';
  }

  private get requestConfig() {
    return {
      adapter: 'http' as const,
      headers: this.QDRANT_API_KEY ? { 'api-key': this.QDRANT_API_KEY } : {},
    };
  }

  generateChunkId(originalId: string, chunkIndex: number) {
    const NAMESPACE = 'a7e6482f-6c26-4cf6-bde3-7151fc4a0a94';
    return uuidV5(`${originalId}_${chunkIndex}`, NAMESPACE);
  }

  async createCollection(collection: string) {
    const collections = await axios.get(`${this.QDRANT_URL}/collections`, this.requestConfig);
    const exists = collections.data?.result?.collections?.some((c: any) => c.name === collection);
    if (!exists) {
      try {
        const res = await axios.put(
          `${this.QDRANT_URL}/collections/${collection}`,
          { vectors: { size: this.EMBEDDING_DIM, distance: 'Cosine' } },
          this.requestConfig,
        );
        return res.data?.result ?? res.data;
      } catch (error: any) {
        if (error?.response?.status === 409) return false;
        throw error;
      }
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
      const res = await axios.put(
        `${this.QDRANT_URL}/collections/${collection}/points?wait=true`,
        { points },
        this.requestConfig,
      );
      return res.data?.result ?? res.data;
    } catch (error) {
      console.error('Qdrant upsert error:', error);
      throw error;
    }
  }

  async search(collection: string, options: SearchQdrantDto) {
    try {
      const res = await axios.post(
        `${this.QDRANT_URL}/collections/${collection}/points/search`,
        options,
        this.requestConfig,
      );
      return res.data?.result ?? [];
    } catch (error: any) {
      // A collection that doesn't exist yet returns 404 — treat as no results.
      if (
        error?.response?.status === 404 ||
        error?.status === 404 ||
        /not found|doesn't exist/i.test(error?.response?.data?.status?.error || error?.message || '')
      ) {
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
        const res = await axios.post(
          `${this.QDRANT_URL}/collections/${collection}/points/scroll`,
          {
            filter,
            limit: pageSize,
            offset,
            with_payload: true,
            with_vector: false,
          },
          this.requestConfig,
        );
        const result = res.data?.result;
        points.push(...(result?.points ?? []));
        offset = result?.next_page_offset;
      } while (offset !== null && offset !== undefined);
    } catch (error: any) {
      console.error('Qdrant scrollAll error details:', {
        status: error?.response?.status || error?.status,
        message: error?.response?.statusText || error?.message,
        data: error?.response?.data || error?.data,
      });
      if (error?.response?.status === 404 || error?.status === 404) return [];
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
        this.requestConfig,
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

  async deletePoints(collection: string, filter: any) {
    try {
      const res = await axios.post(
        `${this.QDRANT_URL}/collections/${collection}/points/delete?wait=true`,
        { filter },
        this.requestConfig,
      );
      return res.data;
    } catch (error: any) {
      console.error('Qdrant deletePoints error details:', {
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
      this.requestConfig,
    );
    return res.data;
  }
}
