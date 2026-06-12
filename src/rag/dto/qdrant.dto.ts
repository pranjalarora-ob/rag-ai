import { IsArray, IsNumber, IsOptional, IsString, ValidateNested, IsObject, IsBoolean } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

class FilterConditionDto {
  @IsString()
  key: string;

  @IsObject()
  match: { value: string | number | boolean };
}

class FilterDto {
  @IsOptional()
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => FilterConditionDto)
  must?: FilterConditionDto[];

  @IsOptional()
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => FilterConditionDto)
  should?: FilterConditionDto[];
}

export class SearchQdrantDto {
  @IsArray()
  @IsNumber({}, { each: true })
  vector: number[];

  @IsOptional()
  @ValidateNested()
  @Type(() => FilterDto)
  filter?: FilterDto;

  @IsOptional()
  @IsNumber()
  limit?: number = 10;

  @IsOptional()
  @IsBoolean()
  with_payload?: boolean = true;

  @IsOptional()
  @IsBoolean()
  with_vector?: boolean = false;

  @IsOptional()
  score_threshold?: number;
}

export type Point = {
  id: number | string;
  vector: number[];
  payload?: Record<string, any>;
};

export class AddSchemaIndexDto {
  @ApiProperty({ example: 'customerId' })
  field: string;

  @ApiProperty({ example: 'keyword' })
  schema: 'keyword' | 'integer' | 'float' | 'geo' | 'text' | 'bool' | 'datetime' | 'uuid';
}
