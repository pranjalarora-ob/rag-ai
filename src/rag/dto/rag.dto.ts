import { IsArray, IsObject, IsString, ValidateNested, IsOptional } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

class DocumentMetadata {
  [key: string]: any;
}

class DocumentDto {
  @ApiProperty({ description: 'Unique identifier for the document' })
  @IsString()
  id: string;

  @ApiProperty({ description: 'Text content of the document' })
  @IsString()
  text: string;

  @ApiProperty({ description: 'Metadata associated with the document', type: Object })
  @IsObject()
  metadata: DocumentMetadata;
}

export class IngestVectorDataDto {
  @ApiProperty({ description: 'Collection name where vectors will be ingested' })
  @IsString()
  collection: string;

  @ApiProperty({ description: 'Documents to be ingested', type: [DocumentDto] })
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => DocumentDto)
  documents: DocumentDto[];

  @ApiProperty({ description: 'Customer ID' })
  @IsString()
  @IsOptional()
  customerId?: string;
}

export class ChatDto {
  @ApiProperty({ description: 'Question to ask the system' })
  @IsString()
  question: string;

  @ApiProperty({ description: 'Customer identifier' })
  @IsString()
  customerId: string;
}

export class PlannerDto {
  @ApiProperty({ description: 'Question to ask the planner' })
  @IsString()
  question: string;

  @ApiProperty({ description: 'Customer identifier' })
  @IsString()
  customerId: string;
}
