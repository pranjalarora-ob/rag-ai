import { SourceDefinition } from '../source.types';
import { projectSource } from './project.source';
import { boqSource } from './boq.source';
import { projectFlowDetailSource } from './project-flow-detail.source';

/** Registry of all ingestable sources, keyed by name (used in the API path). */
export const SOURCES: Record<string, SourceDefinition> = {
  [projectSource.name]: projectSource,
  [boqSource.name]: boqSource,
  [projectFlowDetailSource.name]: projectFlowDetailSource,
};
